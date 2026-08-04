# AI Smart Speaker (Raspberry Pi 5)

ローカルRAG構成のAIスマートスピーカー。ハルシネーションを防ぐため、生成前RAG（検索結果を事前にプロンプトに含める）を採用。

## アーキテクチャ

1. **[耳] listener.py**: Whisper.cpp でユーザーの音声をテキスト化
2. **[検索] retriever.py**: 質問をトリガーに、ローカルの「SearXNG」でWeb検索を実行
3. **[並べ替え] retriever.py**: 日本語Reranker（Optimum/ONNX のクロスエンコーダ）で検索結果を質問との関連度順に並べ替え
4. **[構成] composer.py**: Reranking後の検索結果（事実ソース）と質問をプロンプトに編成
5. **[脳] brain.py**: プロンプトを Ollama（Qwen2.5:3b）に投入し、事実に基づく回答を生成
6. **[口] speaker.py**: `piper-tts-plus` で音声合成して発話
7. **[視覚] status_led.py**: GPIO接続のLEDで、待機・聞き取り・検索・思考・発話・エラーを表示
8. **[操作] push_to_talk.py**: GPIO接続のボタンを押している間だけ録音するプッシュ・トゥ・トーク
9. **[記憶] conversation_history.py**: 直近N回の問いと答えを保持し、再復唱と文脈を踏まえた深掘り質問に対応

## 必要依存ライブラリ

依存ライブラリは `pyproject.toml` で管理しています。

### 本番ロジック用
```bash
pip install .
```

### 開発・テスト用
```bash
pip install -e ".[dev]"
```

### Raspberry Pi（GPIO を使う場合）
```bash
pip install ".[pi]"
```

既存の仮想環境を使う場合:
```bash
source venv/bin/activate
pip install -e ".[dev]"
```

## セットアップ

1. **環境変数の設定**
   ```bash
   cp .env.example .env
   # .envファイルを編集して環境に合わせて設定
   ```

2. **Dockerコンテナの起動**
   ```bash
   # SearXNG
   docker run -d -p 8080:8080 --name searxng searxng/searxng
   
   # Ollama (ローカル実行の場合は不要)
   docker run -d -p 11434:11434 --name ollama ollama/ollama
   ```

3. **音声モデルの配置**

   Piper TTS の音声モデル（日本語ONNX）とその設定ファイルを `models/piper/` に配置します。
   ```bash
   mkdir -p models/piper
   # models/piper/tsukuyomi.onnx      … 音声モデル本体
   # models/piper/tsukuyomi.onnx.json … 対応する設定ファイル
   ```

   設定ファイルは Piper の慣例に従い、モデル名に `.json` を付けた名前にします。
   別のモデルを使う場合は `.env` の `PIPER_MODEL_PATH` / `PIPER_CONFIG_PATH` で指定してください。

4. **Rerankingモデルの配置**

   Rerankingモデルを `models/reranker/` に配置します。
   インターネット非接続環境では、インターネット接続のある別マシンで `optimum-cli` をインストールし、ONNXモデルをエクスポートします。
   ```bash
   pip install optimum[onnxruntime] transformers fugashi unidic-lite
   optimum-cli export onnx \
     --model hotchpotch/japanese-reranker-cross-encoder-xsmall-v1 \
     --task text-classification \
     models/reranker
   ```

   エクスポートした `models/reranker/` ディレクトリを、そのままRaspberry Piのプロジェクトルートへコピーしてください。
   `models/` は `.gitignore` で除外されているため、モデルファイルはGit管理対象になりません。

## 実行

相対インポートを使用しているため、モジュール形式（`-m`）で実行する必要があります。

```bash
# 仮想環境をアクティベート
source venv/bin/activate
python3 -m src.main

# またはシステムPythonを使用
python3 -m src.main
```

## テスト

### ソフトウェアテスト

ユニットテストを実行します：
```bash
python3 -m pytest tests/ -v
```

### ハードウェアの物理動作確認

Raspberry Piに接続したLEDやボタンが物理的に正しく配線されているか確認するためのテストモジュールです。ボタンを押している間LEDが点灯します：
```bash
python3 src/button_led_test.py
```

## ディレクトリ構成

```
smart-speaker/
├── AGENTS.md               # AIエージェント向け開発ルール
├── project_context.md      # プロジェクト要件
├── future_extensions.md    # 将来の機能拡張バックログ
├── README.md               # 本ファイル（セットアップ・実行方法）
├── pyproject.toml          # 依存ライブラリの定義
├── .gitignore              # Git除外設定
├── .env.example            # 環境変数テンプレート
├── models/
│   └── reranker/           # オフラインキャッシュ用Rerankingモデル（.gitignore対象）
├── conftest.py             # src/ を sys.path に追加（テスト用）
├── src/
│   ├── __init__.py
│   ├── main.py             # オーケストレーター
│   ├── listener.py         # Whisper音声認識
│   ├── retriever.py        # SearXNG Web検索
│   ├── composer.py         # RAGプロンプト構成
│   ├── brain.py            # Ollama AI生成
│   ├── speaker.py          # Piper-Plus TTS
│   ├── status_led.py       # GPIO ステータスLED制御
│   ├── push_to_talk.py     # GPIOボタンによるプッシュ・トゥ・トーク
│   ├── conversation_history.py  # マルチターン対話の履歴管理
│   ├── button_led_test.py  # LED・ボタンの物理配線確認用
│   ├── config.py           # 環境変数の一元管理・URL検証
│   ├── http_client.py      # 共通HTTPクライアント
│   ├── exceptions.py       # ドメイン固有の例外
│   └── audio_utils.py      # 音声・ログ共通ユーティリティ
└── tests/
    ├── __init__.py
    ├── conftest.py         # pytest共通フィクスチャ
    ├── test_main.py
    ├── test_listener.py
    ├── test_retriever.py
    ├── test_composer.py
    ├── test_brain.py
    ├── test_speaker.py
    ├── test_status_led.py
    ├── test_push_to_talk.py
    ├── test_conversation_history.py
    ├── test_config.py
    ├── test_http_client.py
    └── test_audio_utils.py
```

## 注意点

- **speaker.py**: 本家Piper（espeak-ng依存）は日本語のアクセント解析が未対応のため、必ず `piper-tts-plus` を使用すること
- **retriever.py（Reranking）**: 標準構成として既定で有効。依存ライブラリ未導入時や処理に失敗した場合は自動的にスキップされ、検索結果をそのまま使用する
- **status_led.py**: `gpiozero` は Raspberry Pi 上でのみ動作するため、非Pi環境では自動的に無効化される
- **conversation_history.py**: 直近 `CONVERSATION_MAX_TURNS` 回分の問いと答えを保持し、要約してプロンプトに埋め込む。「もう一回言って」などの再復唱コマンドは、検索・生成を行わず直前の回答をそのまま発話する
- **push_to_talk.py**: GPIOボタンが利用できる環境では「押している間だけ録音」するプッシュ・トゥ・トークで動作する。ボタンが無い非Pi環境（`gpiozero` 不在）では自動的に無効化され、`RECORD_SECONDS` の固定秒数録音にフォールバックする。`PTT_MIN_RECORD_SECONDS` / `PTT_MAX_RECORD_SECONDS` で最小・最大録音時間を制御する
- **テスト実行時**: 外部API（SearXNG/Ollama）にリクエストを飛ばさず、`pytest-mock` でモック化すること
