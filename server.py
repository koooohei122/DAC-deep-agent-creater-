"""
DAC Local AI Software - HTTP Server
Pure Python, no external dependencies.
Serves the frontend and provides REST API for:
  - Dataset upload & analysis
  - Model configuration & training (with SSE live progress)
  - Model download (JSON)
  - Prediction
  - Feature explanations
"""
import http.server
import json
import os
import sys
import time
import threading
import uuid
import csv
import io
import urllib.parse
import cgi
import math
from pathlib import Path

# ── project imports ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from ai.network import NeuralNetwork
from ai.matrix import Matrix
from ai.data_editor import analyze_and_advise, apply_fixes

# ── Global state ─────────────────────────────────────────────────────────
UPLOADS_DIR = ROOT / "uploads"
MODELS_DIR  = ROOT / "models"
UPLOADS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# session_id -> {model, status, history, config, dataset_id}
SESSIONS: dict = {}
SESSIONS_LOCK = threading.Lock()

# dataset_id -> {filename, path, analysis}
DATASETS: dict = {}

# training event queues: session_id -> list[str]
TRAIN_EVENTS: dict = {}


# ======================================================================== #
# Helpers
# ======================================================================== #
def _ok(data=None, status=200):
    body = json.dumps({"ok": True, "data": data or {}}).encode()
    return status, "application/json", body

def _err(msg, status=400):
    body = json.dumps({"ok": False, "error": msg}).encode()
    return status, "application/json", body

def _load_csv(path) -> Matrix | None:
    """Load CSV -> Matrix (numeric columns only)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None, []
    headers = list(rows[0].keys())
    data = []
    for row in rows:
        try:
            data.append([float(row[h]) for h in headers])
        except ValueError:
            pass
    if not data:
        return None, headers
    return Matrix.from_2d(data), headers

def _split(X: Matrix, y: Matrix, val_ratio=0.2):
    n = X.rows
    split = int(n * (1 - val_ratio))
    Xtr = Matrix.from_2d(X.data[:split])
    ytr = Matrix.from_2d(y.data[:split])
    Xv  = Matrix.from_2d(X.data[split:])
    yv  = Matrix.from_2d(y.data[split:])
    return Xtr, ytr, Xv, yv

def _parse_body(handler):
    length = int(handler.headers.get("Content-Length", 0))
    return json.loads(handler.rfile.read(length)) if length else {}

def _parse_multipart(handler):
    """Parse multipart/form-data -> (fields dict, files dict)."""
    ct = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in ct:
        return {}, {}
    length = int(handler.headers.get("Content-Length", 0))
    raw = handler.rfile.read(length)
    # Extract boundary
    boundary = None
    for part in ct.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[9:].strip('"')
    if not boundary:
        return {}, {}
    sep = ("--" + boundary).encode()
    end = ("--" + boundary + "--").encode()
    parts = raw.split(sep)
    files = {}
    fields = {}
    for p in parts:
        if not p or p == b"--\r\n" or p.strip() == b"--":
            continue
        if p.startswith(b"--"):
            continue
        try:
            header_raw, body = p.split(b"\r\n\r\n", 1)
        except ValueError:
            continue
        body = body.rstrip(b"\r\n")
        if body == b"--":
            continue
        headers_str = header_raw.decode("utf-8", errors="ignore")
        name = filename = None
        for line in headers_str.splitlines():
            if "Content-Disposition" in line:
                for token in line.split(";"):
                    token = token.strip()
                    if token.startswith("name="):
                        name = token[5:].strip('"')
                    if token.startswith("filename="):
                        filename = token[9:].strip('"')
        if name is None:
            continue
        if filename:
            files[name] = (filename, body)
        else:
            fields[name] = body.decode("utf-8", errors="ignore")
    return fields, files


# ======================================================================== #
# Request Handler
# ======================================================================== #
class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"  [{self.address_string()}] " + fmt % args)

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path
        query  = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._serve_file(ROOT / "static" / "index.html", "text/html")
        elif path == "/style.css":
            self._serve_file(ROOT / "static" / "style.css", "text/css")
        elif path == "/app.js":
            self._serve_file(ROOT / "static" / "app.js", "application/javascript")
        elif path == "/api/datasets":
            self._api_list_datasets()
        elif path == "/api/dataset/preview":
            did = query.get("id", [None])[0]
            self._api_dataset_preview(did)
        elif path == "/api/autoconfig":
            did = query.get("id", [None])[0]
            col = query.get("target", [None])[0]
            self._api_autoconfig(did, col)
        elif path == "/api/explain":
            topic = query.get("topic", [""])[0]
            s, ct, b = _ok({"topic": topic, "explanation": _get_explanation(topic)})
            self._respond(s, ct, b)
        elif path == "/api/session":
            sid = query.get("id", [None])[0]
            self._api_get_session(sid)
        elif path == "/api/model/download":
            sid = query.get("id", [None])[0]
            self._api_download_model(sid)
        elif path.startswith("/api/train/events"):
            sid = query.get("id", [None])[0]
            self._api_train_sse(sid)
        elif path == "/api/sample":
            name = query.get("name", ["house_price"])[0]
            self._api_load_sample(name)
        elif path == "/api/session/evaluate":
            sid = query.get("id", [None])[0]
            self._api_evaluate(sid)
        elif path == "/api/session/importance":
            sid = query.get("id", [None])[0]
            self._api_feature_importance(sid)
        else:
            self._respond(404, "text/plain", b"Not found")

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/api/upload":
            self._api_upload()
        elif path == "/api/dataset/fix":
            self._api_fix_dataset()
        elif path == "/api/analyze":
            self._api_analyze()
        elif path == "/api/train/start":
            self._api_train_start()
        elif path == "/api/train/stop":
            self._api_train_stop()
        elif path == "/api/predict":
            self._api_predict()
        elif path == "/api/explain":
            self._api_explain()
        else:
            self._respond(404, "text/plain", b"Not found")

    # ------------------------------------------------------------------ #
    # API endpoints
    # ------------------------------------------------------------------ #
    def _api_upload(self):
        fields, files = _parse_multipart(self)
        if "file" not in files:
            s, ct, b = _err("ファイルが見つかりません")
            self._respond(s, ct, b); return
        filename, content = files["file"]
        did = str(uuid.uuid4())[:8]
        save_path = UPLOADS_DIR / f"{did}_{filename}"
        with open(save_path, "wb") as f:
            f.write(content)
        text = content.decode("utf-8-sig", errors="replace")
        analysis = analyze_and_advise(text)
        DATASETS[did] = {
            "id": did,
            "filename": filename,
            "path": str(save_path),
            "analysis": analysis,
            "size": len(content),
        }
        s, ct, b = _ok({"dataset_id": did, "filename": filename,
                         "analysis": analysis})
        self._respond(s, ct, b)

    def _api_analyze(self):
        body = _parse_body(self)
        did = body.get("dataset_id")
        if did not in DATASETS:
            s, ct, b = _err("データセットが見つかりません"); self._respond(s, ct, b); return
        analysis = DATASETS[did]["analysis"]
        s, ct, b = _ok(analysis)
        self._respond(s, ct, b)

    def _api_fix_dataset(self):
        body = _parse_body(self)
        did = body.get("dataset_id")
        if did not in DATASETS:
            s, ct, b = _err("データセットが見つかりません"); self._respond(s, ct, b); return
        src = DATASETS[did]
        try:
            with open(src["path"], encoding="utf-8-sig") as f:
                csv_text = f.read()
            new_csv, log = apply_fixes(csv_text)
            # Save as new dataset
            new_did = str(uuid.uuid4())[:8]
            base = src["filename"].rsplit(".", 1)[0]
            new_filename = f"{base}_fixed.csv"
            save_path = UPLOADS_DIR / f"{new_did}_{new_filename}"
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(new_csv)
            analysis = analyze_and_advise(new_csv)
            DATASETS[new_did] = {
                "id": new_did,
                "filename": new_filename,
                "path": str(save_path),
                "analysis": analysis,
                "size": len(new_csv.encode()),
            }
            s, ct, b = _ok({
                "new_dataset_id": new_did,
                "filename": new_filename,
                "analysis": analysis,
                "log": log,
            })
        except Exception as e:
            s, ct, b = _err(f"修正処理エラー: {e}")
        self._respond(s, ct, b)

    def _api_load_sample(self, name):
        safe = {"house_price": "house_price.csv", "iris": "iris_numeric.csv"}
        fname = safe.get(name)
        if not fname:
            s, ct, b = _err("サンプル名が無効です"); self._respond(s, ct, b); return
        sample_path = ROOT / "sample_data" / fname
        if not sample_path.exists():
            s, ct, b = _err("サンプルファイルが見つかりません"); self._respond(s, ct, b); return
        content = sample_path.read_bytes()
        text = content.decode("utf-8-sig", errors="replace")
        # Check if already loaded (same filename)
        for ds in DATASETS.values():
            if ds["filename"] == fname:
                s, ct, b = _ok({"dataset_id": ds["id"], "filename": fname, "analysis": ds["analysis"]})
                self._respond(s, ct, b); return
        did = str(uuid.uuid4())[:8]
        save_path = UPLOADS_DIR / f"{did}_{fname}"
        with open(save_path, "wb") as f:
            f.write(content)
        analysis = analyze_and_advise(text)
        DATASETS[did] = {
            "id": did,
            "filename": fname,
            "path": str(save_path),
            "analysis": analysis,
            "size": len(content),
        }
        s, ct, b = _ok({"dataset_id": did, "filename": fname, "analysis": analysis})
        self._respond(s, ct, b)

    def _api_evaluate(self, sid):
        with SESSIONS_LOCK:
            sess = SESSIONS.get(sid)
        if not sess:
            s, ct, b = _err("セッションが見つかりません"); self._respond(s, ct, b); return
        Xv = sess.get("Xv")
        yv = sess.get("yv")
        if Xv is None or yv is None:
            s, ct, b = _err("検証データがありません"); self._respond(s, ct, b); return
        try:
            nn = sess["model"]
            pred = nn.predict(Xv)
            cfg = sess["config"]
            last_act = cfg["layers"][-1]["activation"] if cfg.get("layers") else "linear"
            actuals_flat  = [row[0] for row in yv.data]
            preds_flat    = [row[0] for row in pred.data]

            if last_act == "softmax":
                # Multiclass: argmax
                pred_classes   = [p_row.index(max(p_row)) for p_row in pred.data]
                actual_classes = [int(round(a)) for a in actuals_flat]
                n_classes = max(max(pred_classes), max(actual_classes)) + 1
                matrix = [[0]*n_classes for _ in range(n_classes)]
                for a, p in zip(actual_classes, pred_classes):
                    if 0 <= a < n_classes and 0 <= p < n_classes:
                        matrix[a][p] += 1
                correct = sum(1 for a, p in zip(actual_classes, pred_classes) if a == p)
                acc = correct / max(len(actual_classes), 1)
                result = {"type": "classification", "confusion_matrix": matrix,
                          "accuracy": round(acc, 4), "n_classes": n_classes}
            elif last_act == "sigmoid":
                # Binary classification
                pred_classes   = [1 if v >= 0.5 else 0 for v in preds_flat]
                actual_classes = [int(round(a)) for a in actuals_flat]
                matrix = [[0,0],[0,0]]
                for a, p in zip(actual_classes, pred_classes):
                    if a in (0,1) and p in (0,1):
                        matrix[a][p] += 1
                correct = sum(1 for a, p in zip(actual_classes, pred_classes) if a == p)
                acc = correct / max(len(actual_classes), 1)
                result = {"type": "binary", "confusion_matrix": matrix,
                          "accuracy": round(acc, 4), "n_classes": 2}
            else:
                # Regression: actual vs predicted
                mae = sum(abs(a - p) for a, p in zip(actuals_flat, preds_flat)) / max(len(actuals_flat), 1)
                ss_res = sum((a - p)**2 for a, p in zip(actuals_flat, preds_flat))
                mean_a = sum(actuals_flat) / max(len(actuals_flat), 1)
                ss_tot = sum((a - mean_a)**2 for a in actuals_flat)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
                result = {
                    "type": "regression",
                    "actuals":     [round(v, 4) for v in actuals_flat[:200]],
                    "predictions": [round(v, 4) for v in preds_flat[:200]],
                    "mae": round(mae, 4),
                    "r2":  round(r2, 4),
                }
            s, ct, b = _ok(result)
        except Exception as e:
            s, ct, b = _err(str(e))
        self._respond(s, ct, b)

    def _api_feature_importance(self, sid):
        with SESSIONS_LOCK:
            sess = SESSIONS.get(sid)
        if not sess:
            s, ct, b = _err("セッションが見つかりません"); self._respond(s, ct, b); return
        Xv = sess.get("Xv")
        yv = sess.get("yv")
        feature_cols = sess.get("feature_cols", [])
        if Xv is None or yv is None:
            s, ct, b = _err("検証データがありません"); self._respond(s, ct, b); return
        try:
            import random as _random
            nn = sess["model"]

            def _mse(pred, actual):
                total = sum(
                    (p - a) ** 2
                    for p_row, a_row in zip(pred.data, actual.data)
                    for p, a in zip(p_row, a_row)
                )
                return total / max(len(pred.data), 1)

            baseline_loss = _mse(nn.predict(Xv), yv)
            importances = []
            for i, fname in enumerate(feature_cols):
                shuffled = [row[:] for row in Xv.data]
                col_vals = [row[i] for row in shuffled]
                _random.shuffle(col_vals)
                for j, row in enumerate(shuffled):
                    row[i] = col_vals[j]
                Xv_sh = Matrix.from_2d(shuffled)
                sh_loss = _mse(nn.predict(Xv_sh), yv)
                importances.append({"name": fname, "importance": max(0.0, sh_loss - baseline_loss)})

            total = sum(x["importance"] for x in importances)
            if total > 1e-10:
                for x in importances:
                    x["importance"] = round(x["importance"] / total, 4)
            importances.sort(key=lambda x: -x["importance"])
            s, ct, b = _ok({"feature_importances": importances})
        except Exception as e:
            s, ct, b = _err(str(e))
        self._respond(s, ct, b)

    def _api_dataset_preview(self, did):
        if did not in DATASETS:
            s, ct, b = _err("データセットが見つかりません"); self._respond(s, ct, b); return
        path = DATASETS[did]["path"]
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)[:10]
            if not rows:
                s, ct, b = _err("データが空です"); self._respond(s, ct, b); return
            headers = list(rows[0].keys())
            data = [[row.get(h, "") for h in headers] for row in rows]
            s, ct, b = _ok({"headers": headers, "rows": data, "total": DATASETS[did]["analysis"].get("rows", 0)})
        except Exception as e:
            s, ct, b = _err(str(e))
        self._respond(s, ct, b)

    def _api_autoconfig(self, did, target_col):
        if did not in DATASETS:
            s, ct, b = _err("データセットが見つかりません"); self._respond(s, ct, b); return
        analysis = DATASETS[did]["analysis"]
        col_summary = {c["name"]: c for c in analysis.get("column_summary", [])}
        if not target_col or target_col not in col_summary:
            s, ct, b = _err("目的変数が見つかりません"); self._respond(s, ct, b); return
        col = col_summary[target_col]
        n_features = analysis["columns"] - 1
        n_rows = analysis["rows"]

        # Determine task type
        if col["type"] == "数値":
            task = "regression"
            output_activation = "linear"
            loss = "mse"
            task_label = "回帰"
        else:
            unique = col.get("unique", 2)
            if unique <= 2:
                task = "binary"
                output_activation = "sigmoid"
                loss = "bce"
                task_label = "2値分類"
            else:
                task = "multiclass"
                output_activation = "softmax"
                loss = "cross_entropy"
                task_label = f"多クラス分類 ({unique}クラス)"

        # Suggest architecture based on data size
        if n_rows < 100:
            hidden = [16, 8]
        elif n_rows < 500:
            hidden = [32, 16]
        elif n_rows < 2000:
            hidden = [64, 32]
        else:
            hidden = [128, 64, 32]

        # Adjust for feature count
        hidden = [max(h, n_features * 2) for h in hidden]

        config = {
            "task": task,
            "task_label": task_label,
            "hidden_layers": hidden,
            "activation": "relu",
            "output_activation": output_activation,
            "loss": loss,
            "optimizer": "adam",
            "lr": 0.001,
            "epochs": 50 if n_rows < 500 else 30,
            "batch_size": min(32, max(8, n_rows // 10)),
        }
        reason = (
            f"「{target_col}」は{col['type']}型のため{task_label}タスクと判定。"
            f" データ{n_rows}件・{n_features}特徴量に基づき"
            f" 隠れ層{hidden}を提案。"
        )
        s, ct, b = _ok({"config": config, "reason": reason})
        self._respond(s, ct, b)

    def _api_list_datasets(self):
        lst = [{"id": d["id"], "filename": d["filename"],
                "rows": d["analysis"].get("rows", 0),
                "cols": d["analysis"].get("columns", 0)}
               for d in DATASETS.values()]
        s, ct, b = _ok(lst)
        self._respond(s, ct, b)

    def _api_train_start(self):
        body = _parse_body(self)
        did    = body.get("dataset_id")
        config = body.get("config", {})
        target_col = body.get("target_column")

        if did not in DATASETS:
            s, ct, b = _err("データセットが見つかりません"); self._respond(s, ct, b); return

        ds_path = DATASETS[did]["path"]
        headers = DATASETS[did]["analysis"].get("headers", [])
        if not target_col or target_col not in headers:
            s, ct, b = _err("目的変数を選択してください"); self._respond(s, ct, b); return

        # Load data
        all_mat, all_headers = _load_csv(ds_path)
        if all_mat is None:
            s, ct, b = _err(
                "数値データが読み込めません。文字列列が含まれている場合は「データ」タブの"
                "「✨ すべて自動修正する」ボタンでデータを変換してから再試行してください。"
            ); self._respond(s, ct, b); return

        # Pre-training validation
        if all_mat.rows < 10:
            s, ct, b = _err(
                f"データが少なすぎます（{all_mat.rows} 行）。最低 10 行以上必要です。"
            ); self._respond(s, ct, b); return

        try:
            t_idx = all_headers.index(target_col)
        except ValueError:
            s, ct, b = _err(f"列 '{target_col}' が見つかりません"); self._respond(s, ct, b); return

        feat_idx = [i for i in range(len(all_headers)) if i != t_idx]
        X = Matrix.from_2d([[row[i] for i in feat_idx] for row in all_mat.data])
        y = Matrix.from_2d([[row[t_idx]] for row in all_mat.data])

        # Check target is not constant
        y_vals = [row[0] for row in y.data]
        if len(set(y_vals)) < 2:
            s, ct, b = _err(
                f"目的変数「{target_col}」の値がすべて同じです。別の列を選択してください。"
            ); self._respond(s, ct, b); return

        in_size = X.cols
        out_size = y.cols

        # Build layer config
        hidden = config.get("hidden_layers", [16, 8])
        activation = config.get("activation", "relu")
        last_act = config.get("output_activation", "linear")
        loss_fn = config.get("loss", "mse")
        optimizer = config.get("optimizer", "adam")
        lr = config.get("lr", 0.001)
        epochs = config.get("epochs", 50)
        batch_size = config.get("batch_size", 32)

        layers = []
        prev = in_size
        for h in hidden:
            layers.append({"in": prev, "out": h, "activation": activation})
            prev = h
        layers.append({"in": prev, "out": out_size, "activation": last_act})

        # Validate compatibility
        if last_act == "softmax" and loss_fn not in ("cross_entropy",):
            s, ct, b = _err("Softmax出力にはCross Entropy損失を使用してください")
            self._respond(s, ct, b); return
        if last_act == "sigmoid" and loss_fn not in ("bce", "mse"):
            s, ct, b = _err("Sigmoid出力にはBCEまたはMSE損失を使用してください")
            self._respond(s, ct, b); return

        nn_config = {
            "layers": layers,
            "loss": loss_fn,
            "optimizer": optimizer,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
        }
        nn = NeuralNetwork(nn_config)

        sid = str(uuid.uuid4())[:8]
        # Split before storing so Xv/yv are available for evaluation
        Xtr, ytr, Xv, yv = _split(X, y)

        with SESSIONS_LOCK:
            SESSIONS[sid] = {
                "model": nn,
                "status": "training",
                "config": nn_config,
                "dataset_id": did,
                "target_col": target_col,
                "feature_cols": [all_headers[i] for i in feat_idx],
                "Xv": Xv,
                "yv": yv,
            }
            TRAIN_EVENTS[sid] = []

        def _callback(epoch, loss, val_loss, acc):
            evt = {
                "epoch": epoch,
                "total_epochs": epochs,
                "loss": round(loss, 6),
                "val_loss": round(val_loss, 6) if val_loss is not None else None,
                "accuracy": round(acc, 4),
            }
            with SESSIONS_LOCK:
                if sid in TRAIN_EVENTS:
                    TRAIN_EVENTS[sid].append(evt)

        def _train():
            try:
                nn.train(Xtr, ytr, Xv, yv, callback=_callback)
            except Exception as e:
                print(f"Training error: {e}")
            with SESSIONS_LOCK:
                if sid in SESSIONS:
                    SESSIONS[sid]["status"] = "done"
                if sid in TRAIN_EVENTS:
                    TRAIN_EVENTS[sid].append({"done": True})

        t = threading.Thread(target=_train, daemon=True)
        t.start()

        s, ct, b = _ok({"session_id": sid, "config": nn_config,
                         "features": [all_headers[i] for i in feat_idx],
                         "target": target_col})
        self._respond(s, ct, b)

    def _api_train_sse(self, sid):
        """Server-Sent Events stream for live training progress."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "http://localhost:8080")
        self.end_headers()

        sent = 0
        timeout = 300  # 5 min max
        start = time.time()
        try:
            while time.time() - start < timeout:
                with SESSIONS_LOCK:
                    events = TRAIN_EVENTS.get(sid, [])
                    new_events = events[sent:]
                if new_events:
                    for evt in new_events:
                        data = "data: " + json.dumps(evt) + "\n\n"
                        self.wfile.write(data.encode())
                    self.wfile.flush()
                    sent += len(new_events)
                    if new_events[-1].get("done"):
                        break
                else:
                    # heartbeat
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _api_train_stop(self):
        body = _parse_body(self)
        sid = body.get("session_id")
        with SESSIONS_LOCK:
            sess = SESSIONS.get(sid)
        if sess:
            sess["model"].stop()
            s, ct, b = _ok({"stopped": True})
        else:
            s, ct, b = _err("セッションが見つかりません")
        self._respond(s, ct, b)

    def _api_get_session(self, sid):
        with SESSIONS_LOCK:
            sess = SESSIONS.get(sid)
        if not sess:
            s, ct, b = _err("セッションが見つかりません"); self._respond(s, ct, b); return
        nn = sess["model"]
        s, ct, b = _ok({
            "session_id": sid,
            "status": sess["status"],
            "config": sess["config"],
            "history": nn.history,
            "summary": nn.summary(),
            "features": sess.get("feature_cols", []),
            "target": sess.get("target_col"),
        })
        self._respond(s, ct, b)

    def _api_download_model(self, sid):
        with SESSIONS_LOCK:
            sess = SESSIONS.get(sid)
        if not sess:
            self._respond(404, "text/plain", b"Not found"); return
        nn = sess["model"]
        data = json.dumps(nn.to_dict(), indent=2).encode()
        fname = f"model_{sid}.json"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _api_predict(self):
        body = _parse_body(self)
        sid = body.get("session_id")
        inputs = body.get("inputs", [])
        with SESSIONS_LOCK:
            sess = SESSIONS.get(sid)
        if not sess:
            s, ct, b = _err("セッションが見つかりません"); self._respond(s, ct, b); return
        try:
            X = Matrix.from_2d([inputs] if not isinstance(inputs[0], list) else inputs)
            pred = sess["model"].predict(X)
            s, ct, b = _ok({"predictions": pred.data})
        except Exception as e:
            s, ct, b = _err(str(e))
        self._respond(s, ct, b)

    def _api_explain(self):
        body = _parse_body(self)
        topic = body.get("topic", "")
        explanations = _get_explanation(topic)
        s, ct, b = _ok({"topic": topic, "explanation": explanations})
        self._respond(s, ct, b)

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def _serve_file(self, path: Path, content_type: str):
        if not path.exists():
            self._respond(404, "text/plain", b"Not found"); return
        data = path.read_bytes()
        self._respond(200, content_type, data)

    def _respond(self, status, content_type, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "http://localhost:8080")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "http://localhost:8080")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ======================================================================== #
# Feature explanations
# ======================================================================== #
EXPLANATIONS = {
    "neural_network": """
**ニューラルネットワーク（Neural Network）**
人間の脳の神経細胞（ニューロン）を模倣した計算モデルです。

構造:
- **入力層**: データを受け取る
- **隠れ層**: 特徴を抽出・変換する（複数可）
- **出力層**: 予測値を出力する

各ニューロンは「重み×入力 + バイアス」を計算し、活性化関数を通して次の層に伝えます。
学習とは、この「重み」を調整して予測誤差を最小化する作業です。
""",

    "backpropagation": """
**バックプロパゲーション（誤差逆伝播法）**
ニューラルネットワークの学習アルゴリズムです。

手順:
1. **フォワードパス**: 入力から出力を計算
2. **損失計算**: 予測値と正解の誤差を計算
3. **バックパス**: 連鎖律（Chain Rule）で各重みの勾配を逆方向に計算
4. **パラメータ更新**: 勾配に基づいて重みを更新

このサイクルを繰り返すことで、モデルが"学習"します。
""",

    "activation": """
**活性化関数（Activation Function）**
ニューラルネットワークに非線形性を加える関数です。

主な種類:
- **ReLU**: max(0, x) - 最もよく使われる。勾配消失に強い
- **Sigmoid**: 1/(1+e^-x) - [0,1]に変換。2値分類の出力層に使用
- **Tanh**: [-1,1]に変換。Sigmoidより中心化されている
- **Softmax**: 多クラス分類の出力層で確率分布に変換

線形関数だけだと「深い」ネットワークの意味がなくなります。
活性化関数があることで複雑なパターンを学習できます。
""",

    "loss": """
**損失関数（Loss Function）**
モデルの予測がどれだけ間違っているかを測る関数です。

主な種類:
- **MSE（平均二乗誤差）**: 回帰問題に使用。外れ値に敏感
- **MAE（平均絶対誤差）**: 回帰問題。外れ値にロバスト
- **BCE（2値交差エントロピー）**: 2値分類に使用
- **Cross Entropy（交差エントロピー）**: 多クラス分類に使用

損失が小さいほどモデルが「正解に近い」状態です。
学習はこの損失を最小化する過程です。
""",

    "optimizer": """
**オプティマイザ（Optimizer）**
損失関数を最小化するためにパラメータを更新する手法です。

主な種類:
- **SGD（確率的勾配降下法）**: シンプルだが収束が遅い場合あり
- **SGD + Momentum**: 慣性を持たせて振動を抑制
- **Adam**: 適応的学習率。ほとんどの場合で安定して高速に収束

更新式（SGD）: W = W - lr × ∂L/∂W
""",

    "normalization": """
**正規化・標準化（Normalization / Standardization）**
特徴量のスケールを揃えることで学習を安定させます。

方法:
- **Min-Max正規化**: (x - min) / (max - min) → [0, 1]に変換
- **標準化（Z-score）**: (x - μ) / σ → 平均0、標準偏差1に変換

なぜ必要か:
スケールが異なると、大きな値の特徴量が学習を支配し、
小さな値の特徴量が無視されてしまいます。
""",

    "overfitting": """
**過学習（Overfitting）**
訓練データに「暗記」してしまい、未知データで性能が下がる現象です。

症状:
- 訓練損失は低いが、検証損失が高い

対策:
- **正則化**: L1/L2 ペナルティを損失に追加
- **Dropout**: 学習中にランダムにニューロンを無効化
- **データ拡張**: 学習データを人工的に増やす
- **早期停止（Early Stopping）**: 検証損失が悪化したら学習を止める
- **モデルを単純にする**: 隠れ層や neurons を減らす
""",

    "batch": """
**バッチサイズ（Batch Size）**
1回のパラメータ更新に使用するデータ件数です。

種類:
- **バッチ学習（全データ）**: 安定だが遅く、メモリ大
- **ミニバッチ（一般的）**: バランスが良い（32〜256が一般的）
- **オンライン学習（1件）**: 速いが不安定

小さいバッチ:
- ノイズが多いが汎化性能が上がることも
大きいバッチ:
- 安定だが過学習しやすく、鋭いミニマムに収束しやすい
""",
}

def _get_explanation(topic: str) -> str:
    topic_lower = topic.lower().strip()
    # keyword matching
    mapping = {
        "neural": "neural_network",
        "ニューラル": "neural_network",
        "network": "neural_network",
        "backprop": "backpropagation",
        "逆伝播": "backpropagation",
        "誤差逆": "backpropagation",
        "activation": "activation",
        "活性化": "activation",
        "relu": "activation",
        "sigmoid": "activation",
        "loss": "loss",
        "損失": "loss",
        "mse": "loss",
        "optimizer": "optimizer",
        "最適化": "optimizer",
        "adam": "optimizer",
        "sgd": "optimizer",
        "normal": "normalization",
        "正規化": "normalization",
        "標準化": "normalization",
        "overfit": "overfitting",
        "過学習": "overfitting",
        "batch": "batch",
        "バッチ": "batch",
    }
    for kw, key in mapping.items():
        if kw in topic_lower:
            return EXPLANATIONS[key]

    # Return all topics list
    topics = "\n".join([f"- **{k}**: 解説あり" for k in EXPLANATIONS.keys()])
    return f"トピック「{topic}」の解説が見つかりません。\n\n利用可能なトピック:\n{topics}"


# ======================================================================== #
# Entry point
# ======================================================================== #
def main():
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "127.0.0.1")
    server = http.server.ThreadingHTTPServer((host, port), Handler)
    print("=" * 60)
    print("  DAC Local AI Software")
    print(f"  http://localhost:{port}")
    print("=" * 60)
    print("  ブラウザで上記URLを開いてください。")
    print("  Ctrl+C で停止")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止しました。")


if __name__ == "__main__":
    main()
