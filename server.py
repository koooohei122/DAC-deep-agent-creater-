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
from ai.data_editor import analyze_and_advise

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
        elif path == "/api/session":
            sid = query.get("id", [None])[0]
            self._api_get_session(sid)
        elif path == "/api/model/download":
            sid = query.get("id", [None])[0]
            self._api_download_model(sid)
        elif path.startswith("/api/train/events"):
            sid = query.get("id", [None])[0]
            self._api_train_sse(sid)
        else:
            self._respond(404, "text/plain", b"Not found")

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/api/upload":
            self._api_upload()
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
            s, ct, b = _err("数値データが読み込めません"); self._respond(s, ct, b); return

        try:
            t_idx = all_headers.index(target_col)
        except ValueError:
            s, ct, b = _err(f"列 '{target_col}' が見つかりません"); self._respond(s, ct, b); return

        feat_idx = [i for i in range(len(all_headers)) if i != t_idx]
        X = Matrix.from_2d([[row[i] for i in feat_idx] for row in all_mat.data])
        y = Matrix.from_2d([[row[t_idx]] for row in all_mat.data])

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
        with SESSIONS_LOCK:
            SESSIONS[sid] = {
                "model": nn,
                "status": "training",
                "config": nn_config,
                "dataset_id": did,
                "target_col": target_col,
                "feature_cols": [all_headers[i] for i in feat_idx],
            }
            TRAIN_EVENTS[sid] = []

        # Train in background thread
        Xtr, ytr, Xv, yv = _split(X, y)

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
        self.send_header("Access-Control-Allow-Origin", "*")
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
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
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
    server = http.server.ThreadingHTTPServer(("", port), Handler)
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
