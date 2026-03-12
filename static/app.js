/* ============================================================
   DAC Local AI - Frontend Application
   Pure Vanilla JS, no frameworks or external libraries
   ============================================================ */

'use strict';

// ================================================================
// State
// ================================================================
const state = {
  datasets: {},          // id -> dataset info
  sessions: {},          // id -> session info
  activeDataset: null,
  activeSession: null,
  trainSSE: null,        // EventSource for live training
};

// ================================================================
// Tab navigation
// ================================================================
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');

    if (btn.dataset.tab === 'preview') refreshPreview();
    if (btn.dataset.tab === 'predict') refreshPredictSessionList();
  });
});

// ================================================================
// Upload Tab
// ================================================================
const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const uploadStat = document.getElementById('upload-status');

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('over');
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
});

async function uploadFile(file) {
  if (!file.name.endsWith('.csv')) {
    showStatus('upload-status', '❌ CSVファイルのみ対応しています', 'error');
    return;
  }
  showStatus('upload-status', '⏳ アップロード中…', '');
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res  = await fetch('/api/upload', { method: 'POST', body: fd });
    const json = await res.json();
    if (!json.ok) throw new Error(json.error);
    const { dataset_id, filename, analysis } = json.data;
    state.datasets[dataset_id] = { id: dataset_id, filename, analysis };
    state.activeDataset = dataset_id;
    showStatus('upload-status', `✅ ${filename} をアップロードしました (${analysis.rows} 行, ${analysis.columns} 列)`, 'success');
    renderDatasetList();
    renderAnalysis(dataset_id);
    refreshTrainDatasets();
  } catch (e) {
    showStatus('upload-status', `❌ ${e.message}`, 'error');
  }
}

function showStatus(id, msg, type) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.className = `upload-status ${type}`;
  el.classList.remove('hidden');
}

function renderDatasetList() {
  const list = document.getElementById('dataset-list');
  list.innerHTML = '';
  const datasets = Object.values(state.datasets);
  if (!datasets.length) {
    list.innerHTML = '<p class="empty-hint">まだデータがありません</p>';
    return;
  }
  datasets.forEach(ds => {
    const item = document.createElement('div');
    item.className = `dataset-item${ds.id === state.activeDataset ? ' selected' : ''}`;
    item.innerHTML = `
      <div>
        <div class="dataset-name">📄 ${esc(ds.filename)}</div>
        <div class="dataset-meta">${ds.analysis.rows} 行 × ${ds.analysis.columns} 列</div>
      </div>
      <div class="dataset-meta">${ds.id}</div>`;
    item.addEventListener('click', () => {
      state.activeDataset = ds.id;
      renderDatasetList();
      renderAnalysis(ds.id);
    });
    list.appendChild(item);
  });
}

function renderAnalysis(did) {
  const ds = state.datasets[did];
  if (!ds) return;
  const a = ds.analysis;
  const section = document.getElementById('analysis-section');
  section.classList.remove('hidden');

  document.getElementById('analysis-meta').innerHTML =
    `${a.rows} 行 &nbsp;|&nbsp; ${a.columns} 列 &nbsp;|&nbsp;
     <span class="text-high">${a.high_priority} 件 HIGH</span> &nbsp;/&nbsp;
     ${a.total_issues} 件 合計`;

  // Column table
  const tbody = document.getElementById('col-tbody');
  tbody.innerHTML = '';
  (a.column_summary || []).forEach(col => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><code>${esc(col.name)}</code></td>
      <td>${col.type}</td>
      <td>${col.nulls}</td>
      <td class="${col.null_rate !== '0.0%' ? 'text-medium' : ''}">${col.null_rate}</td>
      <td>${col.min ?? '—'}</td>
      <td>${col.max ?? '—'}</td>
      <td>${col.mean ?? (col.unique ?? '—')}</td>
      <td>${col.std ?? (col.samples ? col.samples.slice(0,3).join(', ') : '—')}</td>
      <td class="${col.outliers > 0 ? 'text-medium' : ''}">${col.outliers ?? '—'}</td>`;
    tbody.appendChild(tr);
  });

  // Instructions
  const instrSec  = document.getElementById('instructions-section');
  const instrList = document.getElementById('instructions-list');
  const instrCount= document.getElementById('instruction-count');
  instrCount.textContent = a.total_issues;
  instrList.innerHTML = '';
  if (!a.instructions || !a.instructions.length) {
    instrSec.innerHTML += '<p class="empty-hint">問題は検出されませんでした 🎉</p>';
    return;
  }
  a.instructions.forEach(inst => {
    const div = document.createElement('div');
    div.className = `instruction-item ${inst.priority}`;
    div.innerHTML = `
      <div class="instr-header">
        <span class="priority-badge ${inst.priority}">${inst.priority}</span>
        <span class="instr-col">${esc(inst.column)}</span>
        <span class="instr-issue">— ${esc(inst.issue)}</span>
      </div>
      <div class="instr-action">🔧 ${esc(inst.action)}</div>
      <div class="instr-why">💡 ${esc(inst.why)}</div>`;
    instrList.appendChild(div);
  });
}

// ================================================================
// Train Tab
// ================================================================
const trainDatasetSel = document.getElementById('train-dataset');
const trainTargetSel  = document.getElementById('train-target');

function refreshTrainDatasets() {
  const cur = trainDatasetSel.value;
  trainDatasetSel.innerHTML = '<option value="">-- 選択 --</option>';
  Object.values(state.datasets).forEach(ds => {
    const opt = document.createElement('option');
    opt.value = ds.id; opt.textContent = ds.filename;
    if (ds.id === cur) opt.selected = true;
    trainDatasetSel.appendChild(opt);
  });
}

trainDatasetSel.addEventListener('change', () => {
  const did = trainDatasetSel.value;
  trainTargetSel.innerHTML = '<option value="">-- 選択 --</option>';
  if (!did || !state.datasets[did]) return;
  const headers = state.datasets[did].analysis.headers || [];
  headers.forEach(h => {
    const opt = document.createElement('option');
    opt.value = h; opt.textContent = h;
    trainTargetSel.appendChild(opt);
  });
  // Guess target = last column
  if (headers.length) trainTargetSel.value = headers[headers.length - 1];

  // Show summary preview
  renderModelSummaryPreview();
});

['hidden-layers','activation','output-activation','loss-fn','optimizer','lr','epochs','batch-size']
  .forEach(id => document.getElementById(id).addEventListener('change', renderModelSummaryPreview));

function renderModelSummaryPreview() {
  const did = trainDatasetSel.value;
  if (!did || !state.datasets[did]) return;
  const ds = state.datasets[did];
  const target = trainTargetSel.value;
  if (!target) return;

  const headers  = ds.analysis.headers || [];
  const features = headers.filter(h => h !== target);
  const inSize   = features.length;
  const outSize  = 1;

  const hiddenStr = document.getElementById('hidden-layers').value;
  const hidden = hiddenStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
  const act    = document.getElementById('activation').value;
  const outAct = document.getElementById('output-activation').value;

  let layers = [[`入力`, inSize]];
  hidden.forEach((h, i) => layers.push([`隠れ ${i+1} (${act})`, h]));
  layers.push([`出力 (${outAct})`, outSize]);

  const totalParams = () => {
    let total = 0, prev = inSize;
    hidden.forEach(h => { total += prev * h + h; prev = h; });
    total += prev * outSize + outSize;
    return total;
  };

  const sumDiv = document.getElementById('model-summary');
  sumDiv.innerHTML = `
    <table>
      <thead><tr><th>層</th><th>ユニット数</th><th>パラメータ</th></tr></thead>
      <tbody>${layers.map((l, i) => {
        const prev  = i === 0 ? 0 : (layers[i-1][1]);
        const params = i === 0 ? '—' : (prev * l[1] + l[1]).toLocaleString();
        return `<tr><td>${esc(l[0])}</td><td>${l[1]}</td><td>${params}</td></tr>`;
      }).join('')}</tbody>
    </table>
    <div class="total-params">合計パラメータ: ${totalParams().toLocaleString()}</div>
    <div style="margin-top:.5rem;font-size:.78rem;color:var(--text2)">
      入力特徴量: ${esc(features.join(', '))}<br>
      目的変数: <span style="color:var(--accent)">${esc(target)}</span>
    </div>`;
}

// Train
document.getElementById('btn-train').addEventListener('click', startTraining);
document.getElementById('btn-stop').addEventListener('click', stopTraining);

async function startTraining() {
  const did = trainDatasetSel.value;
  if (!did) { alert('データセットを選択してください'); return; }
  const target = trainTargetSel.value;
  if (!target) { alert('目的変数を選択してください'); return; }

  const hiddenStr = document.getElementById('hidden-layers').value;
  const hidden = hiddenStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);

  const config = {
    hidden_layers:       hidden,
    activation:          document.getElementById('activation').value,
    output_activation:   document.getElementById('output-activation').value,
    loss:                document.getElementById('loss-fn').value,
    optimizer:           document.getElementById('optimizer').value,
    lr:                  parseFloat(document.getElementById('lr').value),
    epochs:              parseInt(document.getElementById('epochs').value),
    batch_size:          parseInt(document.getElementById('batch-size').value),
  };

  document.getElementById('btn-train').disabled = true;
  document.getElementById('btn-stop').disabled  = false;
  document.getElementById('train-status-bar').classList.remove('hidden');
  updateProgress(0, 'セッション開始中…');

  try {
    const res  = await fetch('/api/train/start', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ dataset_id: did, config, target_column: target }),
    });
    const json = await res.json();
    if (!json.ok) throw new Error(json.error);

    const sid = json.data.session_id;
    state.sessions[sid] = { id: sid, status: 'training', history: { loss: [], val_loss: [], accuracy: [] } };
    state.activeSession = sid;
    refreshPreviewSessionList();
    refreshPredictSessionList();
    startSSE(sid, config.epochs);
  } catch (e) {
    alert(`エラー: ${e.message}`);
    trainDone();
  }
}

function startSSE(sid, totalEpochs) {
  if (state.trainSSE) state.trainSSE.close();
  const sse = new EventSource(`/api/train/events?id=${sid}`);
  state.trainSSE = sse;

  sse.onmessage = e => {
    const evt = JSON.parse(e.data);
    if (evt.done) {
      state.sessions[sid].status = 'done';
      updateProgress(100, '学習完了 ✅');
      trainDone();
      sse.close();
      // Switch to preview
      setTimeout(() => {
        document.querySelector('[data-tab="preview"]').click();
      }, 800);
      return;
    }
    const { epoch, loss, val_loss, accuracy } = evt;
    const pct = Math.round((epoch / totalEpochs) * 100);
    const vlStr = val_loss != null ? `  val_loss: ${val_loss.toFixed(4)}` : '';
    updateProgress(pct,
      `Epoch ${epoch}/${totalEpochs} &nbsp;|&nbsp; loss: ${loss.toFixed(4)}${vlStr} &nbsp;|&nbsp; acc: ${(accuracy*100).toFixed(1)}%`);

    // Update session history
    const sess = state.sessions[sid];
    sess.history.loss.push(loss);
    if (val_loss != null) sess.history.val_loss.push(val_loss);
    sess.history.accuracy.push(accuracy);

    // If preview tab is active, live-draw
    if (document.getElementById('tab-preview').classList.contains('active')) {
      drawCharts(sid);
    }
  };

  sse.onerror = () => {
    trainDone();
    sse.close();
  };
}

function stopTraining() {
  if (!state.activeSession) return;
  fetch('/api/train/stop', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ session_id: state.activeSession }),
  });
}

function trainDone() {
  document.getElementById('btn-train').disabled = false;
  document.getElementById('btn-stop').disabled  = true;
}

function updateProgress(pct, msg) {
  document.getElementById('progress-bar').style.width = `${pct}%`;
  document.getElementById('train-status-text').innerHTML = msg;
}

// ================================================================
// Preview Tab
// ================================================================
function refreshPreviewSessionList() {
  const sel = document.getElementById('preview-session');
  const cur = sel.value;
  sel.innerHTML = '<option value="">-- 選択 --</option>';
  Object.keys(state.sessions).forEach(sid => {
    const opt = document.createElement('option');
    opt.value = sid; opt.textContent = `Session ${sid}`;
    if (sid === cur || sid === state.activeSession) opt.selected = true;
    sel.appendChild(opt);
  });
  const active = sel.value;
  if (active) document.getElementById('btn-download').disabled = false;
}

document.getElementById('preview-session').addEventListener('change', () => {
  const sid = document.getElementById('preview-session').value;
  if (sid) {
    document.getElementById('btn-download').disabled = false;
    drawCharts(sid);
    drawArchitecture(sid);
  }
});

document.getElementById('btn-download').addEventListener('click', () => {
  const sid = document.getElementById('preview-session').value;
  if (sid) window.location = `/api/model/download?id=${sid}`;
});

function refreshPreview() {
  refreshPreviewSessionList();
  const sid = document.getElementById('preview-session').value || state.activeSession;
  if (sid && state.sessions[sid]) {
    drawCharts(sid);
    drawArchitecture(sid);
  }
}

// ----------------------------------------------------------------
// Canvas charts
// ----------------------------------------------------------------
function drawCharts(sid) {
  const sess = state.sessions[sid];
  if (!sess) return;
  drawLineChart('chart-loss',
    [sess.history.loss, sess.history.val_loss.length ? sess.history.val_loss : null],
    ['#58a6ff', '#f0883e'],
    ['訓練損失', '検証損失']);
  drawLineChart('chart-acc',
    [sess.history.accuracy],
    ['#7ee787'],
    ['精度']);
}

function drawLineChart(canvasId, seriesList, colors, labels) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || canvas.width;
  const H = canvas.height;
  canvas.width = W;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#21262d';
  ctx.fillRect(0, 0, W, H);

  const pad = { top: 16, right: 20, bottom: 32, left: 52 };
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top - pad.bottom;

  // Find value range
  let minV = Infinity, maxV = -Infinity;
  seriesList.forEach(s => {
    if (!s) return;
    s.forEach(v => { if (v < minV) minV = v; if (v > maxV) maxV = v; });
  });
  if (minV === maxV) { minV -= 0.1; maxV += 0.1; }
  const vRange = maxV - minV || 1;

  const maxLen = Math.max(...seriesList.filter(Boolean).map(s => s.length));
  if (maxLen < 2) return;

  // Grid lines
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + ch * (i / 4);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y);
    ctx.stroke();
    const v = maxV - (vRange * i / 4);
    ctx.fillStyle = '#8b949e'; ctx.font = '11px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(4), pad.left - 6, y + 4);
  }

  // X-axis labels
  ctx.fillStyle = '#8b949e'; ctx.font = '11px monospace'; ctx.textAlign = 'center';
  for (let i = 0; i <= 4; i++) {
    const x = pad.left + cw * (i / 4);
    const ep = Math.round(maxLen * i / 4);
    ctx.fillText(ep, x, H - 8);
  }

  // Series
  seriesList.forEach((s, si) => {
    if (!s || s.length < 2) return;
    ctx.beginPath(); ctx.strokeStyle = colors[si]; ctx.lineWidth = 2;
    s.forEach((v, i) => {
      const x = pad.left + (i / (maxLen - 1)) * cw;
      const y = pad.top + (1 - (v - minV) / vRange) * ch;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Legend
    const lx = pad.left + si * 110 + 8;
    const ly = pad.top + 12;
    ctx.fillStyle = colors[si];
    ctx.fillRect(lx, ly - 7, 14, 4);
    ctx.fillStyle = '#c9d1d9'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(labels[si], lx + 18, ly);
  });
}

function drawArchitecture(sid) {
  const sess = state.sessions[sid];
  if (!sess || !sess.config) return;
  const canvas = document.getElementById('chart-arch');
  if (!canvas) return;
  const W = canvas.offsetWidth || 900;
  canvas.width = W;
  const H = 200;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#21262d'; ctx.fillRect(0, 0, W, H);

  const layers = sess.config.layers;
  if (!layers) return;

  const totalLayers = layers.length + 1; // +1 for input
  const colW = W / totalLayers;
  const inSize = layers[0].in;

  // Draw columns: input + each layer's output
  const cols = [{ label: '入力', size: inSize }];
  layers.forEach((l, i) => {
    cols.push({ label: `L${i+1}\n${l.activation}`, size: l.out });
  });

  const maxDots = 6;
  const dotR = 7;
  const cx = (i) => colW * i + colW / 2;

  // Draw connections first
  ctx.lineWidth = 0.6;
  for (let ci = 0; ci < cols.length - 1; ci++) {
    const a = cols[ci], b = cols[ci + 1];
    const aDots = Math.min(a.size, maxDots);
    const bDots = Math.min(b.size, maxDots);
    ctx.strokeStyle = 'rgba(88,166,255,0.12)';
    for (let ai = 0; ai < aDots; ai++) {
      const ay = dotY(ai, aDots, H);
      for (let bi = 0; bi < bDots; bi++) {
        const by = dotY(bi, bDots, H);
        ctx.beginPath();
        ctx.moveTo(cx(ci) + dotR, ay);
        ctx.lineTo(cx(ci + 1) - dotR, by);
        ctx.stroke();
      }
    }
  }

  // Draw nodes
  cols.forEach((col, ci) => {
    const n = Math.min(col.size, maxDots);
    const isLast = ci === cols.length - 1;
    for (let ni = 0; ni < n; ni++) {
      const y = dotY(ni, n, H);
      ctx.beginPath();
      ctx.arc(cx(ci), y, dotR, 0, Math.PI * 2);
      ctx.fillStyle = isLast ? '#7ee787' : (ci === 0 ? '#f0883e' : '#58a6ff');
      ctx.fill();
    }
    if (col.size > maxDots) {
      ctx.fillStyle = '#8b949e'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('⋮', cx(ci), H / 2 + 4);
    }
    // Label
    ctx.fillStyle = '#c9d1d9'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
    const lines = col.label.split('\n');
    lines.forEach((line, li) => {
      ctx.fillText(line, cx(ci), H - 14 + li * 13);
    });
    // Size badge
    ctx.fillStyle = '#484f58'; ctx.font = 'bold 10px monospace';
    ctx.fillText(`×${col.size}`, cx(ci), 16);
  });
}

function dotY(i, total, H) {
  const pad = 30;
  const range = H - pad * 2;
  return total === 1 ? H / 2 : pad + (i / (total - 1)) * range;
}

// ================================================================
// Predict Tab
// ================================================================
function refreshPredictSessionList() {
  ['predict-session'].forEach(id => {
    const sel = document.getElementById(id);
    const cur = sel.value;
    sel.innerHTML = '<option value="">-- 選択 --</option>';
    Object.keys(state.sessions).forEach(sid => {
      const opt = document.createElement('option');
      opt.value = sid; opt.textContent = `Session ${sid}`;
      if (sid === cur || sid === state.activeSession) opt.selected = true;
      sel.appendChild(opt);
    });
    if (sel.value) buildPredictForm(sel.value);
  });
}

document.getElementById('predict-session').addEventListener('change', () => {
  buildPredictForm(document.getElementById('predict-session').value);
});

function buildPredictForm(sid) {
  const container = document.getElementById('predict-inputs');
  const btn = document.getElementById('btn-predict');
  if (!sid || !state.sessions[sid]) {
    container.innerHTML = '<p class="empty-hint">セッションを選択してください</p>';
    btn.disabled = true;
    return;
  }
  const features = state.sessions[sid].features || [];
  if (!features.length) {
    container.innerHTML = '<p class="empty-hint">特徴量情報がありません</p>';
    return;
  }
  container.innerHTML = features.map(f => `
    <div class="predict-field">
      <label>${esc(f)}</label>
      <input type="number" step="any" data-feature="${esc(f)}" placeholder="0.0">
    </div>`).join('');
  btn.disabled = false;
}

document.getElementById('btn-predict').addEventListener('click', async () => {
  const sid = document.getElementById('predict-session').value;
  if (!sid) return;
  const inputs = [...document.querySelectorAll('#predict-inputs input')]
    .map(el => parseFloat(el.value) || 0);

  try {
    const res  = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ session_id: sid, inputs }),
    });
    const json = await res.json();
    if (!json.ok) throw new Error(json.error);
    const preds = json.data.predictions;
    const val   = preds[0][0];
    const target = state.sessions[sid].target || '予測値';
    document.getElementById('predict-result').innerHTML = `
      <div class="result-label">${esc(target)}</div>
      <div class="result-value">${typeof val === 'number' ? val.toFixed(6) : val}</div>
      <div style="margin-top:.75rem;font-size:.8rem;color:var(--text2)">
        全出力: [${preds[0].map(v => v.toFixed(4)).join(', ')}]
      </div>`;
  } catch (e) {
    document.getElementById('predict-result').innerHTML =
      `<p style="color:var(--danger)">エラー: ${esc(e.message)}</p>`;
  }
});

// ================================================================
// Explain Tab
// ================================================================
document.querySelectorAll('.topic-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.topic-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    fetchExplain(btn.dataset.topic);
  });
});

document.getElementById('btn-explain').addEventListener('click', () => {
  const q = document.getElementById('explain-input').value.trim();
  if (q) fetchExplain(q);
});
document.getElementById('explain-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') fetchExplain(e.target.value.trim());
});

async function fetchExplain(topic) {
  try {
    const res  = await fetch('/api/explain', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ topic }),
    });
    const json = await res.json();
    if (!json.ok) throw new Error(json.error);
    renderExplain(json.data.topic, json.data.explanation);
  } catch (e) {
    document.getElementById('explain-content').textContent = `エラー: ${e.message}`;
  }
}

function renderExplain(topic, text) {
  document.getElementById('explain-title').textContent = `解説: ${topic}`;
  const content = document.getElementById('explain-content');
  // Simple markdown: **bold**, newlines
  const html = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
  content.innerHTML = html;
}

// ================================================================
// Session management - load from server on demand
// ================================================================
async function loadSession(sid) {
  try {
    const res  = await fetch(`/api/session?id=${sid}`);
    const json = await res.json();
    if (!json.ok) return;
    const d = json.data;
    state.sessions[sid] = {
      id: sid,
      status: d.status,
      config: d.config,
      history: d.history,
      features: d.features,
      target: d.target,
    };
  } catch {}
}

// ================================================================
// Utility
// ================================================================
function esc(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

// ================================================================
// Boot
// ================================================================
(async function boot() {
  // Load existing datasets
  try {
    const res  = await fetch('/api/datasets');
    const json = await res.json();
    if (json.ok && json.data.length) {
      json.data.forEach(ds => {
        if (!state.datasets[ds.id]) {
          state.datasets[ds.id] = ds;
        }
      });
      renderDatasetList();
      refreshTrainDatasets();
    }
  } catch {}
})();
