/* ============================================================
   DAC Local AI - Frontend Application  (v2 - intuitive UX)
   Vanilla JS, no external libraries
   ============================================================ */

'use strict';

// ================================================================
// State
// ================================================================
const state = {
  datasets: {},
  sessions: {},
  activeDataset: null,
  activeSession: null,
  trainSSE: null,
};

// ================================================================
// Tab navigation with step locking
// ================================================================
function switchTab(tabName) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
  const btn = document.querySelector(`[data-tab="${tabName}"]`);
  if (btn) btn.classList.add('active');
  const section = document.getElementById(`tab-${tabName}`);
  if (section) section.classList.add('active');
  if (tabName === 'preview') refreshPreview();
  if (tabName === 'predict') refreshPredictSessionList();
}

function unlockTab(tabName) {
  const btn = document.querySelector(`[data-tab="${tabName}"]`);
  if (btn) {
    btn.removeAttribute('disabled');
    btn.querySelector('.step-done')?.classList.remove('hidden');
  }
}

function markTabDone(tabName) {
  const btn = document.querySelector(`[data-tab="${tabName}"]`);
  btn?.querySelector('.step-done')?.classList.remove('hidden');
}

function setHint(msg, done = false) {
  document.getElementById('hint-text').textContent = msg;
  document.getElementById('workflow-hint').classList.toggle('done', done);
}

document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    if (btn.disabled || btn.getAttribute('disabled') !== null) return;
    switchTab(btn.dataset.tab);
  });
});

// ================================================================
// Upload Tab
// ================================================================
const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');

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
  if (!file.name.toLowerCase().endsWith('.csv')) {
    showStatus('upload-status', '❌ CSVファイルのみ対応しています', 'error'); return;
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
    showStatus('upload-status',
      `✅ ${filename}  (${analysis.rows}行 × ${analysis.columns}列)`, 'success');
    renderDatasetList();
    renderAnalysis(dataset_id);
    await fetchDataPreview(dataset_id);
    refreshTrainDatasets();
    unlockTab('train');
    markTabDone('upload');
    setHint(`「${filename}」をアップロードしました。次は「学習」タブでモデルを設定してください →`);
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
    list.innerHTML = `<div class="empty-state">
      <div class="empty-icon">📋</div>
      <p>まだデータがありません</p>
      <p class="hint">左のゾーンにCSVをドロップするか<br>「ファイルを選択」をクリックしてください</p>
    </div>`;
    return;
  }
  datasets.forEach(ds => {
    const item = document.createElement('div');
    item.className = `dataset-item${ds.id === state.activeDataset ? ' selected' : ''}`;
    const hi = ds.analysis.high_priority || 0;
    item.innerHTML = `
      <div>
        <div class="dataset-name">📄 ${esc(ds.filename)}</div>
        <div class="dataset-meta">${ds.analysis.rows}行 × ${ds.analysis.columns}列
          ${hi > 0 ? `&nbsp;<span class="text-high">⚠ HIGH×${hi}</span>` : ''}
        </div>
      </div>
      <button class="btn btn-primary btn-sm" style="flex-shrink:0">
        学習タブへ →
      </button>`;
    item.querySelector('button').addEventListener('click', e => {
      e.stopPropagation();
      state.activeDataset = ds.id;
      renderDatasetList();
      unlockTab('train');
      switchTab('train');
      // Pre-select in train tab
      const sel = document.getElementById('train-dataset');
      sel.value = ds.id;
      sel.dispatchEvent(new Event('change'));
    });
    item.addEventListener('click', () => {
      state.activeDataset = ds.id;
      renderDatasetList();
      renderAnalysis(ds.id);
      fetchDataPreview(ds.id);
    });
    list.appendChild(item);
  });
}

async function fetchDataPreview(did) {
  try {
    const res  = await fetch(`/api/dataset/preview?id=${did}`);
    const json = await res.json();
    if (!json.ok) return;
    const { headers, rows, total } = json.data;
    const section = document.getElementById('data-preview-section');
    const caption = document.getElementById('preview-caption');
    section.classList.remove('hidden');
    caption.textContent = `最初 ${rows.length} 行 / 合計 ${total} 行`;

    const tbl = document.getElementById('preview-table');
    const thead = `<thead><tr>${headers.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead>`;
    const tbody = `<tbody>${rows.map(row =>
      `<tr>${row.map(v => {
        const isEmpty = v === '' || v === null || v === undefined ||
                        String(v).toLowerCase() === 'nan';
        return `<td class="${isEmpty ? 'null-cell' : ''}">${isEmpty ? '—' : esc(String(v))}</td>`;
      }).join('')}</tr>`
    ).join('')}</tbody>`;
    tbl.innerHTML = thead + tbody;
  } catch {}
}

function renderAnalysis(did) {
  const ds = state.datasets[did];
  if (!ds) return;
  const a = ds.analysis;
  document.getElementById('analysis-section').classList.remove('hidden');
  document.getElementById('analysis-meta').innerHTML =
    `${a.rows} 行 &nbsp;|&nbsp; ${a.columns} 列 &nbsp;|&nbsp;
     <span class="text-high">${a.high_priority} 件 HIGH</span> &nbsp;/&nbsp;
     ${a.total_issues} 件 合計`;

  const tbody = document.getElementById('col-tbody');
  tbody.innerHTML = '';
  (a.column_summary || []).forEach(col => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><code>${esc(col.name)}</code></td>
      <td>${col.type}</td>
      <td>${col.nulls}</td>
      <td class="${col.null_rate !== '0.0%' ? 'text-medium' : ''}">${col.null_rate}</td>
      <td>${col.min ?? '—'}</td><td>${col.max ?? '—'}</td>
      <td>${col.mean ?? (col.unique ?? '—')}</td>
      <td>${col.std ?? (col.samples ? col.samples.slice(0,3).join(', ') : '—')}</td>
      <td class="${col.outliers > 0 ? 'text-medium' : ''}">${col.outliers ?? '—'}</td>`;
    tbody.appendChild(tr);
  });

  const instrCount = document.getElementById('instruction-count');
  instrCount.textContent = a.total_issues;
  const instrList = document.getElementById('instructions-list');
  instrList.innerHTML = '';
  if (!a.instructions?.length) {
    instrList.innerHTML = '<p class="empty-hint">問題は検出されませんでした 🎉</p>';
  } else {
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
  document.getElementById('goto-train-cta').classList.remove('hidden');

  // Show fix bar if there are issues
  const fixBar = document.getElementById('fix-bar');
  const fixStatus = document.getElementById('fix-status');
  const fixBtn = document.getElementById('btn-fix-all');
  if (fixBar) {
    if (a.total_issues > 0) {
      fixBar.classList.remove('hidden');
      fixStatus.textContent = '';
      fixStatus.className = 'fix-status';
      if (fixBtn) fixBtn.disabled = false;
    } else {
      fixBar.classList.add('hidden');
    }
  }
}

document.getElementById('btn-goto-train')?.addEventListener('click', () => {
  unlockTab('train');
  switchTab('train');
  const sel = document.getElementById('train-dataset');
  if (state.activeDataset) {
    sel.value = state.activeDataset;
    sel.dispatchEvent(new Event('change'));
  }
});

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
  document.getElementById('btn-autoconfig').disabled = true;
  hideTaskHint();
  if (!did || !state.datasets[did]) return;
  const headers = state.datasets[did].analysis.headers || [];
  headers.forEach(h => {
    const opt = document.createElement('option');
    opt.value = h; opt.textContent = h;
    trainTargetSel.appendChild(opt);
  });
  if (headers.length) {
    trainTargetSel.value = headers[headers.length - 1];
    trainTargetSel.dispatchEvent(new Event('change'));
  }
});

trainTargetSel.addEventListener('change', () => {
  const did   = trainDatasetSel.value;
  const target = trainTargetSel.value;
  if (!did || !target) return;
  document.getElementById('btn-autoconfig').disabled = false;
  autoConfigFromTarget(did, target);
  renderModelSummaryPreview();
});

// Auto-detect task type from column stats
function autoConfigFromTarget(did, target) {
  const ds = state.datasets[did];
  if (!ds) return;
  const col = ds.analysis.column_summary?.find(c => c.name === target);
  if (!col) return;

  let taskClass = '', taskMsg = '';
  if (col.type === '数値') {
    setSelect('output-activation', 'linear');
    setSelect('loss-fn', 'mse');
    taskClass = 'regression';
    taskMsg = '📊 回帰タスクとして自動設定しました（Linear出力 + MSE損失）';
  } else if (col.unique <= 2) {
    setSelect('output-activation', 'sigmoid');
    setSelect('loss-fn', 'bce');
    taskClass = 'binary';
    taskMsg = '✅ 2値分類タスクとして自動設定しました（Sigmoid出力 + BCE損失）';
  } else {
    setSelect('output-activation', 'softmax');
    setSelect('loss-fn', 'cross_entropy');
    taskClass = 'multiclass';
    taskMsg = `🎯 多クラス分類（${col.unique}クラス）として自動設定しました（Softmax出力 + CrossEntropy損失）`;
  }
  showTaskHint(taskClass, taskMsg);
}

function showTaskHint(cls, msg) {
  const el = document.getElementById('task-hint');
  el.className = `task-hint ${cls}`;
  el.textContent = msg;
  el.classList.remove('hidden');
}
function hideTaskHint() {
  document.getElementById('task-hint').classList.add('hidden');
}

function setSelect(id, value) {
  const el = document.getElementById(id);
  if (el) el.value = value;
}

// AI auto-config button
document.getElementById('btn-autoconfig').addEventListener('click', async () => {
  const did    = trainDatasetSel.value;
  const target = trainTargetSel.value;
  if (!did || !target) return;
  try {
    const res  = await fetch(`/api/autoconfig?id=${did}&target=${encodeURIComponent(target)}`);
    const json = await res.json();
    if (!json.ok) throw new Error(json.error);
    const { config, reason } = json.data;
    // Apply
    document.getElementById('hidden-layers').value = config.hidden_layers.join(',');
    setSelect('activation',        config.activation);
    setSelect('output-activation', config.output_activation);
    setSelect('loss-fn',           config.loss);
    setSelect('optimizer',         config.optimizer);
    document.getElementById('lr').value         = config.lr;
    document.getElementById('epochs').value     = config.epochs;
    document.getElementById('batch-size').value = config.batch_size;

    const banner = document.getElementById('autoconfig-banner');
    document.getElementById('autoconfig-title').textContent = `🤖 AIが設定を自動提案しました（${config.task_label}）`;
    document.getElementById('autoconfig-reason').textContent = reason;
    banner.classList.remove('hidden');
    renderModelSummaryPreview();
  } catch (e) {
    alert(`自動設定エラー: ${e.message}`);
  }
});

// Form change -> update summary
['hidden-layers','activation','output-activation','loss-fn','optimizer','lr','epochs','batch-size']
  .forEach(id => document.getElementById(id)?.addEventListener('change', renderModelSummaryPreview));

function renderModelSummaryPreview() {
  const did    = trainDatasetSel.value;
  const target = trainTargetSel.value;
  if (!did || !state.datasets[did] || !target) return;
  const ds = state.datasets[did];
  const headers  = ds.analysis.headers || [];
  const features = headers.filter(h => h !== target);
  const inSize   = features.length;
  const outSize  = 1;
  const hiddenStr = document.getElementById('hidden-layers').value;
  const hidden = hiddenStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
  const act    = document.getElementById('activation').value;
  const outAct = document.getElementById('output-activation').value;

  const cols = [['入力', inSize, '—']];
  let prev = inSize;
  hidden.forEach((h, i) => {
    cols.push([`隠れ ${i+1} (${act})`, h, (prev * h + h).toLocaleString()]);
    prev = h;
  });
  cols.push([`出力 (${outAct})`, outSize, (prev * outSize + outSize).toLocaleString()]);
  const total = hidden.reduce((acc, h, i) => {
    const p = i === 0 ? inSize : hidden[i-1];
    return acc + p * h + h;
  }, 0) + prev * outSize + outSize;

  document.getElementById('model-summary').innerHTML = `
    <table>
      <thead><tr><th>層</th><th>ユニット</th><th>パラメータ</th></tr></thead>
      <tbody>${cols.map(([n, u, p]) =>
        `<tr><td>${esc(n)}</td><td>${u}</td><td>${p}</td></tr>`).join('')}
      </tbody>
    </table>
    <div class="total-params">合計パラメータ: ${total.toLocaleString()}</div>
    <div style="margin-top:.5rem;font-size:.78rem;color:var(--text2)">
      入力特徴量 (${features.length}): ${esc(features.join(', '))}<br>
      目的変数: <span style="color:var(--accent)">${esc(target)}</span>
    </div>`;
}

// Train
document.getElementById('btn-train').addEventListener('click', startTraining);
document.getElementById('btn-stop').addEventListener('click', stopTraining);

async function startTraining() {
  const did    = trainDatasetSel.value;
  const target = trainTargetSel.value;
  if (!did)    { alert('データセットを選択してください'); return; }
  if (!target) { alert('目的変数を選択してください'); return; }

  const hiddenStr = document.getElementById('hidden-layers').value;
  const hidden = hiddenStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
  const epochs = parseInt(document.getElementById('epochs').value);

  const config = {
    hidden_layers:     hidden,
    activation:        document.getElementById('activation').value,
    output_activation: document.getElementById('output-activation').value,
    loss:              document.getElementById('loss-fn').value,
    optimizer:         document.getElementById('optimizer').value,
    lr:                parseFloat(document.getElementById('lr').value),
    epochs,
    batch_size:        parseInt(document.getElementById('batch-size').value),
  };

  document.getElementById('btn-train').disabled = true;
  document.getElementById('btn-stop').disabled  = false;
  document.getElementById('train-status-bar').classList.remove('hidden');
  document.getElementById('train-health-badge').classList.add('hidden');
  document.getElementById('train-metrics').innerHTML = '';
  updateProgress(0, '接続中…');

  try {
    const res  = await fetch('/api/train/start', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ dataset_id: did, config, target_column: target }),
    });
    const json = await res.json();
    if (!json.ok) throw new Error(json.error);
    const sid = json.data.session_id;
    state.sessions[sid] = {
      id: sid, status: 'training', config: json.data.config,
      history: { loss: [], val_loss: [], accuracy: [] },
      features: json.data.features,
      target: json.data.target,
    };
    state.activeSession = sid;
    refreshPreviewSessionList();
    refreshPredictSessionList();
    setHint(`学習中 Session ${sid} — プレビュータブでグラフを確認できます`);
    startSSE(sid, epochs);
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
      updateProgress(100, '学習完了');
      setHealthBadge('good', '学習完了 ✅');
      trainDone();
      markTabDone('train');
      unlockTab('preview');
      unlockTab('predict');
      setHint('学習完了！「プレビュー」でグラフを確認し「予測」で試してみましょう ✓', true);
      sse.close();
      setTimeout(() => switchTab('preview'), 600);
      return;
    }
    const { epoch, loss, val_loss, accuracy } = evt;
    const pct = Math.round((epoch / totalEpochs) * 100);
    updateProgress(pct, `Epoch ${epoch} / ${totalEpochs}`);
    updateMetrics(loss, val_loss, accuracy);

    const sess = state.sessions[sid];
    sess.history.loss.push(loss);
    if (val_loss != null) sess.history.val_loss.push(val_loss);
    sess.history.accuracy.push(accuracy);

    // Health detection
    detectTrainingHealth(sess.history);

    if (document.getElementById('tab-preview').classList.contains('active')) {
      drawCharts(sid);
    }
  };
  sse.onerror = () => { trainDone(); sse.close(); };
}

function detectTrainingHealth(history) {
  const loss    = history.loss;
  const valLoss = history.val_loss;
  const n = loss.length;
  if (n < 8) return;

  const recentLoss = loss.slice(-5);
  const improving = recentLoss[0] - recentLoss[recentLoss.length - 1] > 0.001;

  if (valLoss.length >= 5) {
    const vl = valLoss.slice(-5);
    const overfitting = vl.every((v, i) => i === 0 || v >= vl[i-1]) &&
                        loss[loss.length-1] < loss[loss.length-5] * 0.95;
    if (overfitting) {
      setHealthBadge('danger', '⚠ 過学習の兆候');
      return;
    }
  }
  if (!improving) {
    setHealthBadge('warn', '⏸ 損失が停滞中');
  } else {
    setHealthBadge('good', '↘ 損失改善中');
  }
}

function setHealthBadge(cls, text) {
  const el = document.getElementById('train-health-badge');
  el.className = `health-badge ${cls}`;
  el.textContent = text;
  el.classList.remove('hidden');
}

function updateMetrics(loss, valLoss, accuracy) {
  const container = document.getElementById('train-metrics');
  const vl = valLoss != null ? valLoss.toFixed(4) : '—';
  container.innerHTML = `
    <div class="metric-chip">
      <div class="metric-label">Loss</div>
      <div class="metric-value">${loss.toFixed(4)}</div>
    </div>
    <div class="metric-chip">
      <div class="metric-label">Val Loss</div>
      <div class="metric-value">${vl}</div>
    </div>
    <div class="metric-chip">
      <div class="metric-label">Accuracy</div>
      <div class="metric-value">${(accuracy * 100).toFixed(1)}%</div>
    </div>`;
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
  document.getElementById('train-status-text').textContent = msg;
}

// ================================================================
// Inline tooltips (? buttons)
// ================================================================
document.getElementById('tab-train').addEventListener('click', async e => {
  const btn = e.target.closest('.info-icon');
  if (!btn) return;
  const topic = btn.dataset.topic;
  try {
    const res  = await fetch(`/api/explain?topic=${encodeURIComponent(topic)}`);
    const json = await res.json();
    if (!json.ok) return;
    showTooltip(btn, json.data.explanation);
  } catch {}
});

function showTooltip(anchor, markdown) {
  const tooltip = document.getElementById('inline-tooltip');
  const body    = document.getElementById('tooltip-body');
  body.innerHTML = renderMarkdown(markdown);
  tooltip.classList.remove('hidden');
  // Position near button
  const rect = anchor.getBoundingClientRect();
  const tW = 340, tH = 250;
  let left = rect.left;
  let top  = rect.bottom + 8 + window.scrollY;
  if (left + tW > window.innerWidth - 8) left = window.innerWidth - tW - 8;
  if (left < 8) left = 8;
  tooltip.style.left = `${left}px`;
  tooltip.style.top  = `${top}px`;
}

document.getElementById('tooltip-close').addEventListener('click', () => {
  document.getElementById('inline-tooltip').classList.add('hidden');
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') document.getElementById('inline-tooltip').classList.add('hidden');
});

// ================================================================
// Preview Tab
// ================================================================
function refreshPreviewSessionList() {
  const sel = document.getElementById('preview-session');
  const cur = sel.value || state.activeSession;
  sel.innerHTML = '<option value="">-- 選択 --</option>';
  Object.keys(state.sessions).forEach(sid => {
    const opt = document.createElement('option');
    const sess = state.sessions[sid];
    opt.value = sid;
    opt.textContent = `Session ${sid} (${sess.status === 'done' ? '完了' : '学習中'})`;
    if (sid === cur) opt.selected = true;
    sel.appendChild(opt);
  });
  const active = sel.value;
  if (active) {
    document.getElementById('btn-download').disabled = false;
    drawCharts(active);
    drawArchitecture(active);
    maybeShowEvalSection(active);
  }
}

document.getElementById('preview-session').addEventListener('change', () => {
  const sid = document.getElementById('preview-session').value;
  if (sid) {
    document.getElementById('btn-download').disabled = false;
    drawCharts(sid);
    drawArchitecture(sid);
    maybeShowEvalSection(sid);
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
    ['#58a6ff', '#f0883e'], ['訓練損失', '検証損失']);
  drawLineChart('chart-acc',
    [sess.history.accuracy], ['#7ee787'], ['精度']);
}

function drawLineChart(canvasId, seriesList, colors, labels) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const W = canvas.offsetWidth || canvas.width;
  const H = canvas.height;
  canvas.width = W;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#21262d';
  ctx.fillRect(0, 0, W, H);

  const pad = { top: 20, right: 20, bottom: 32, left: 56 };
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top - pad.bottom;

  let minV = Infinity, maxV = -Infinity;
  seriesList.forEach(s => { if (!s) return; s.forEach(v => { if (v < minV) minV = v; if (v > maxV) maxV = v; }); });
  if (!isFinite(minV)) return;
  if (minV === maxV) { minV -= 0.1; maxV += 0.1; }
  const vRange = maxV - minV;
  const maxLen = Math.max(...seriesList.filter(Boolean).map(s => s.length));
  if (maxLen < 2) return;

  // Grid
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + ch * (i / 4);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke();
    const v = maxV - (vRange * i / 4);
    ctx.fillStyle = '#8b949e'; ctx.font = '11px monospace'; ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(3), pad.left - 5, y + 4);
  }
  // X labels
  ctx.fillStyle = '#8b949e'; ctx.font = '11px monospace'; ctx.textAlign = 'center';
  for (let i = 0; i <= 4; i++) {
    const x = pad.left + cw * (i / 4);
    ctx.fillText(Math.round(maxLen * i / 4), x, H - 6);
  }

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
    ctx.fillStyle = colors[si]; ctx.fillRect(lx, pad.top + 8, 14, 4);
    ctx.fillStyle = '#c9d1d9'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(labels[si], lx + 18, pad.top + 13);
  });
}

function drawArchitecture(sid) {
  const sess = state.sessions[sid];
  if (!sess?.config?.layers) return;
  const canvas = document.getElementById('chart-arch');
  if (!canvas) return;
  const W = canvas.offsetWidth || 900;
  canvas.width = W; canvas.height = 200;
  const H = 200;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#21262d'; ctx.fillRect(0, 0, W, H);

  const layers = sess.config.layers;
  const cols = [{ label: '入力', size: layers[0].in }];
  layers.forEach((l, i) => cols.push({ label: `L${i+1}\n${l.activation}`, size: l.out }));

  const maxDots = 6, dotR = 7;
  const colW = W / cols.length;
  const cx = i => colW * i + colW / 2;

  // Connections
  ctx.lineWidth = 0.5; ctx.strokeStyle = 'rgba(88,166,255,.1)';
  for (let ci = 0; ci < cols.length - 1; ci++) {
    const aN = Math.min(cols[ci].size, maxDots);
    const bN = Math.min(cols[ci+1].size, maxDots);
    for (let ai = 0; ai < aN; ai++) {
      for (let bi = 0; bi < bN; bi++) {
        ctx.beginPath();
        ctx.moveTo(cx(ci) + dotR, dotY(ai, aN, H));
        ctx.lineTo(cx(ci+1) - dotR, dotY(bi, bN, H));
        ctx.stroke();
      }
    }
  }
  // Nodes
  cols.forEach((col, ci) => {
    const n = Math.min(col.size, maxDots);
    const isLast = ci === cols.length - 1;
    for (let ni = 0; ni < n; ni++) {
      ctx.beginPath();
      ctx.arc(cx(ci), dotY(ni, n, H), dotR, 0, Math.PI * 2);
      ctx.fillStyle = isLast ? '#7ee787' : (ci === 0 ? '#f0883e' : '#58a6ff');
      ctx.fill();
    }
    if (col.size > maxDots) {
      ctx.fillStyle = '#8b949e'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('⋮', cx(ci), H / 2 + 4);
    }
    ctx.fillStyle = '#c9d1d9'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
    col.label.split('\n').forEach((line, li) => ctx.fillText(line, cx(ci), H - 14 + li * 13));
    ctx.fillStyle = '#484f58'; ctx.font = 'bold 10px monospace';
    ctx.fillText(`×${col.size}`, cx(ci), 16);
  });
}

function dotY(i, total, H) {
  const pad = 30;
  return total === 1 ? H / 2 : pad + (i / (total - 1)) * (H - pad * 2);
}

// ================================================================
// Predict Tab
// ================================================================
function refreshPredictSessionList() {
  const sel = document.getElementById('predict-session');
  const cur = sel.value || state.activeSession;
  sel.innerHTML = '<option value="">-- 選択 --</option>';
  Object.keys(state.sessions).forEach(sid => {
    const opt = document.createElement('option');
    opt.value = sid;
    opt.textContent = `Session ${sid}`;
    if (sid === cur) opt.selected = true;
    sel.appendChild(opt);
  });
  if (sel.value) buildPredictForm(sel.value);
}

document.getElementById('predict-session').addEventListener('change', () => {
  buildPredictForm(document.getElementById('predict-session').value);
});

function buildPredictForm(sid) {
  const container = document.getElementById('predict-inputs');
  const btn = document.getElementById('btn-predict');
  if (!sid || !state.sessions[sid]) {
    container.innerHTML = `<div class="empty-state">
      <div class="empty-icon">✏️</div>
      <p>セッションを選択してください</p>
      <p class="hint">先に「学習」タブでモデルをトレーニングしてください</p>
    </div>`;
    btn.disabled = true; return;
  }
  const features = state.sessions[sid].features || [];
  if (!features.length) {
    // Try loading from server
    loadSession(sid).then(() => buildPredictForm(sid));
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
    const res  = await fetch(`/api/explain?topic=${encodeURIComponent(topic)}`);
    const json = await res.json();
    if (!json.ok) throw new Error(json.error);
    document.getElementById('explain-title').textContent = `解説: ${json.data.topic}`;
    document.getElementById('explain-content').innerHTML = renderMarkdown(json.data.explanation);
  } catch (e) {
    document.getElementById('explain-content').textContent = `エラー: ${e.message}`;
  }
}

// ================================================================
// Markdown-lite renderer (for explanations)
// ================================================================
function renderMarkdown(text) {
  return text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}

// ================================================================
// Session loader
// ================================================================
async function loadSession(sid) {
  try {
    const res  = await fetch(`/api/session?id=${sid}`);
    const json = await res.json();
    if (!json.ok) return;
    const d = json.data;
    state.sessions[sid] = {
      id: sid, status: d.status, config: d.config,
      history: d.history, features: d.features, target: d.target,
    };
  } catch {}
}

// ================================================================
// Utility
// ================================================================
function esc(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ================================================================
// Sample data quick-start
// ================================================================
document.getElementById('btn-sample-house')?.addEventListener('click', () => loadSample('house_price'));
document.getElementById('btn-sample-iris')?.addEventListener('click',  () => loadSample('iris'));

async function loadSample(name) {
  const label = name === 'house_price' ? '住宅価格' : '花の種類';
  showStatus('upload-status', `⏳ サンプルデータ「${label}」を読み込み中…`, '');
  try {
    const res  = await fetch(`/api/sample?name=${name}`);
    const json = await res.json();
    if (!json.ok) throw new Error(json.error);
    const { dataset_id, filename, analysis } = json.data;
    state.datasets[dataset_id] = { id: dataset_id, filename, analysis };
    state.activeDataset = dataset_id;
    showStatus('upload-status',
      `✅ サンプル「${filename}」を読み込みました（${analysis.rows}行 × ${analysis.columns}列）`, 'success');
    renderDatasetList();
    renderAnalysis(dataset_id);
    await fetchDataPreview(dataset_id);
    refreshTrainDatasets();
    unlockTab('train');
    markTabDone('upload');
    setHint(`サンプルデータを読み込みました。次は「学習」タブでモデルを設定してください →`);
  } catch (e) {
    showStatus('upload-status', `❌ ${e.message}`, 'error');
  }
}

// ================================================================
// Fix dataset
// ================================================================
document.getElementById('btn-fix-all')?.addEventListener('click', fixDataset);

async function fixDataset() {
  const did = state.activeDataset;
  if (!did) return;
  const statusEl = document.getElementById('fix-status');
  const btn = document.getElementById('btn-fix-all');
  btn.disabled = true;
  statusEl.textContent = '⏳ 修正中…';
  try {
    const res  = await fetch('/api/dataset/fix', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ dataset_id: did }),
    });
    const json = await res.json();
    if (!json.ok) throw new Error(json.error);
    const { new_dataset_id, filename, analysis, log } = json.data;
    state.datasets[new_dataset_id] = { id: new_dataset_id, filename, analysis };
    state.activeDataset = new_dataset_id;
    renderDatasetList();
    renderAnalysis(new_dataset_id);
    await fetchDataPreview(new_dataset_id);
    refreshTrainDatasets();
    const summary = log.length ? log.slice(0, 3).join(' / ') + (log.length > 3 ? ` 他${log.length-3}件` : '') : '変更なし';
    statusEl.textContent = `✅ 修正完了: ${summary}`;
    statusEl.className = 'fix-status success';
    showFixDetail(log);
  } catch (e) {
    statusEl.textContent = `❌ ${e.message}`;
    statusEl.className = 'fix-status error';
    btn.disabled = false;
  }
}

function showFixDetail(log) {
  if (!log.length) return;
  const list = document.getElementById('instructions-list');
  const div = document.createElement('div');
  div.className = 'fix-log-card';
  div.innerHTML = `<strong>✨ 自動修正ログ</strong><ul>${log.map(l => `<li>${esc(l)}</li>`).join('')}</ul>`;
  list.insertBefore(div, list.firstChild);
}

// ================================================================
// Evaluation (Preview tab)
// ================================================================
document.getElementById('btn-evaluate')?.addEventListener('click', runEvaluation);

async function runEvaluation() {
  const sid = document.getElementById('preview-session').value;
  if (!sid) return;
  const btn = document.getElementById('btn-evaluate');
  btn.disabled = true;
  btn.textContent = '⏳ 評価中…';
  const content = document.getElementById('eval-content');
  content.innerHTML = '<p class="hint">計算中…</p>';
  try {
    const [evalRes, impRes] = await Promise.all([
      fetch(`/api/session/evaluate?id=${sid}`).then(r => r.json()),
      fetch(`/api/session/importance?id=${sid}`).then(r => r.json()),
    ]);
    content.innerHTML = '';
    if (evalRes.ok) renderEvalResult(content, evalRes.data);
    if (impRes.ok) renderFeatureImportance(content, impRes.data.feature_importances);
  } catch (e) {
    content.innerHTML = `<p style="color:var(--danger)">エラー: ${esc(e.message)}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '📊 評価を実行';
  }
}

function renderEvalResult(container, data) {
  if (data.type === 'regression') {
    const section = document.createElement('div');
    section.className = 'eval-block';
    section.innerHTML = `
      <h4 class="eval-sub">回帰評価 &nbsp;<span class="badge-secondary">MAE: ${data.mae} &nbsp;|&nbsp; R²: ${data.r2}</span></h4>
      <p class="hint">実測値 vs 予測値（検証データ）</p>
      <canvas id="chart-scatter" width="400" height="280"></canvas>`;
    container.appendChild(section);
    setTimeout(() => drawScatter('chart-scatter', data.actuals, data.predictions), 50);
  } else {
    const section = document.createElement('div');
    section.className = 'eval-block';
    const acc = (data.accuracy * 100).toFixed(1);
    section.innerHTML = `
      <h4 class="eval-sub">分類評価 &nbsp;<span class="badge-secondary">精度: ${acc}%</span></h4>
      <p class="hint">混同行列（縦軸: 正解, 横軸: 予測）</p>
      <canvas id="chart-confusion" width="320" height="320"></canvas>`;
    container.appendChild(section);
    setTimeout(() => drawConfusionMatrix('chart-confusion', data.confusion_matrix), 50);
  }
}

function renderFeatureImportance(container, importances) {
  if (!importances?.length) return;
  const section = document.createElement('div');
  section.className = 'eval-block';
  section.innerHTML = `
    <h4 class="eval-sub">特徴重要度 <span class="hint-inline">（値が大きいほど予測への影響が大きい）</span></h4>
    <div id="importance-bars" class="importance-bars"></div>`;
  container.appendChild(section);
  const bars = section.querySelector('#importance-bars');
  const maxVal = importances[0]?.importance || 1;
  importances.forEach(({ name, importance }) => {
    const pct = maxVal > 0 ? (importance / maxVal * 100).toFixed(1) : 0;
    const bar = document.createElement('div');
    bar.className = 'imp-row';
    bar.innerHTML = `
      <span class="imp-name">${esc(name)}</span>
      <div class="imp-bar-wrap">
        <div class="imp-bar" style="width:${pct}%"></div>
      </div>
      <span class="imp-val">${(importance * 100).toFixed(1)}%</span>`;
    bars.appendChild(bar);
  });
}

function drawScatter(canvasId, actuals, predictions) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const W = canvas.offsetWidth || 400;
  canvas.width = W;
  const H = canvas.height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#21262d';
  ctx.fillRect(0, 0, W, H);

  const pad = { top: 20, right: 20, bottom: 40, left: 50 };
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top - pad.bottom;

  const all = [...actuals, ...predictions];
  const mn = Math.min(...all);
  const mx = Math.max(...all);
  const rng = mx - mn || 1;
  const toX = v => pad.left + (v - mn) / rng * cw;
  const toY = v => pad.top + (1 - (v - mn) / rng) * ch;

  // Perfect prediction line
  ctx.strokeStyle = '#444'; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(toX(mn), toY(mn)); ctx.lineTo(toX(mx), toY(mx)); ctx.stroke();
  ctx.setLineDash([]);

  // Points
  ctx.fillStyle = '#58a6ff';
  actuals.forEach((a, i) => {
    ctx.beginPath();
    ctx.arc(toX(a), toY(predictions[i]), 3, 0, Math.PI * 2);
    ctx.fill();
  });

  // Axes labels
  ctx.fillStyle = '#8b949e'; ctx.font = '11px sans-serif';
  ctx.textAlign = 'center'; ctx.fillText('実測値', pad.left + cw / 2, H - 4);
  ctx.save(); ctx.translate(12, pad.top + ch / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('予測値', 0, 0); ctx.restore();
}

function drawConfusionMatrix(canvasId, matrix) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const n = matrix.length;
  const W = canvas.width;
  const H = canvas.height;
  const pad = 30;
  const cellW = (W - pad * 2) / n;
  const cellH = (H - pad * 2) / n;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#21262d'; ctx.fillRect(0, 0, W, H);

  const maxVal = Math.max(...matrix.flat());

  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const v = matrix[r][c];
      const alpha = maxVal > 0 ? v / maxVal : 0;
      const x = pad + c * cellW;
      const y = pad + r * cellH;
      ctx.fillStyle = r === c
        ? `rgba(126,231,135,${0.15 + alpha * 0.75})`
        : `rgba(248,81,73,${alpha * 0.7})`;
      ctx.fillRect(x, y, cellW - 1, cellH - 1);
      ctx.fillStyle = '#c9d1d9';
      ctx.font = `bold ${Math.min(14, Math.floor(cellH * 0.4))}px monospace`;
      ctx.textAlign = 'center';
      ctx.fillText(v, x + cellW / 2, y + cellH / 2 + 5);
    }
  }

  // Axis labels
  ctx.fillStyle = '#8b949e'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
  for (let i = 0; i < n; i++) {
    ctx.fillText(i, pad + i * cellW + cellW / 2, pad - 6);
    ctx.fillText(i, pad - 10, pad + i * cellH + cellH / 2 + 4);
  }
}

// Show evaluation section when a session is selected
function maybeShowEvalSection(sid) {
  const sec = document.getElementById('eval-section');
  if (!sec) return;
  if (sid && state.sessions[sid]?.status === 'done') {
    sec.classList.remove('hidden');
    // Reset content
    document.getElementById('eval-content').innerHTML =
      '<p class="hint">「評価を実行」ボタンを押すと、混同行列・特徴重要度などが表示されます。</p>';
    document.getElementById('btn-evaluate').disabled = false;
    document.getElementById('btn-evaluate').textContent = '📊 評価を実行';
  } else {
    sec.classList.add('hidden');
  }
}

// ================================================================
// Boot
// ================================================================
(async function boot() {
  try {
    const res  = await fetch('/api/datasets');
    const json = await res.json();
    if (json.ok && json.data.length) {
      json.data.forEach(ds => { if (!state.datasets[ds.id]) state.datasets[ds.id] = ds; });
      renderDatasetList();
      refreshTrainDatasets();
      unlockTab('train');
    }
  } catch {}
})();
