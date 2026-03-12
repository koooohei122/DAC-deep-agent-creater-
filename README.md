# DAC Local AI

ローカルで動くAIソフトウェア。**外部ライブラリ不使用**、Python標準ライブラリのみで実装。

## 機能

| 機能 | 説明 |
|------|------|
| 🧮 純Python AIエンジン | 行列演算・NN・バックプロパゲーション・Adam/SGDを自作 |
| 📂 データアップロード | CSVをドラッグ&ドロップでアップロード |
| 🔍 ML編集アドバイザ | 欠損値・外れ値・スケール・歪み・クラス不均衡を自動検出し編集指示を表示 |
| 📈 リアルタイム学習プレビュー | SSEで損失・精度グラフをライブ更新 |
| ⬇ モデルダウンロード | 学習済みモデルをJSONでダウンロード |
| 🔮 予測 | 学習済みモデルで新しいデータを予測 |
| 📖 機能解説 | NN・バックプロパゲーション・活性化関数などをわかりやすく解説 |

## 起動方法

```bash
python server.py
```

ブラウザで `http://localhost:8080` を開く。

## ファイル構成

```
DAC-deep-agent-creater-/
├── server.py              # HTTPサーバー (Python標準ライブラリのみ)
├── ai/
│   ├── matrix.py          # 行列演算 (自作)
│   ├── activations.py     # 活性化関数 (ReLU/Sigmoid/Tanh/Softmax)
│   ├── losses.py          # 損失関数 (MSE/MAE/BCE/CrossEntropy)
│   ├── optimizers.py      # SGD/Adam (自作)
│   ├── network.py         # ニューラルネットワーク本体
│   └── data_editor.py     # MLデータ分析・編集アドバイザ
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── uploads/               # アップロードしたCSV
└── models/                # 保存モデル
```

## 依存関係

**Python 3.9+ のみ** — pip install 不要。
