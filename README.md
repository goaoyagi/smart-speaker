# AI Smart Speaker (Raspberry Pi 5)

ローカル完結型の日本語 AI スマートスピーカー。詳細な要件・設計背景は `project_context.md` を参照。

## ディレクトリ構成

```
smart-speaker/
├── project_context.md      # プロジェクト要件・アーキテクチャ
├── AGENTS.md               # AI エージェント向け開発ルール
├── future_extensions.md    # 将来拡張バックログ
├── README.md               # 本ファイル（セットアップ・実行方法）
├── .env.example            # 環境変数テンプレート
├── conftest.py             # ルート conftest（sys.path 設定）
├── src/
│   ├── main.py             # VoiceAssistant オーケストレーター
│   ├── listener.py         # [耳] Whisper 音声認識
│   ├── retriever.py        # [検索] SearXNG Web 検索
│   ├── composer.py         # [構成] RAG プロンプト編成
│   ├── brain.py            # [脳] Ollama AI 生成
│   ├── speaker.py          # [口] Piper-Plus TTS
│   ├── status_led.py       # [視覚] GPIO LED 制御
│   ├── config.py           # 全環境変数の一元管理
│   ├── exceptions.py       # カスタム例外階層
│   ├── http_client.py      # HTTP 共通ユーティリティ
│   └── audio_utils.py      # 音声ユーティリティ・ログヘルパー
└── tests/
    ├── conftest.py          # 共通フィクスチャ
    ├── test_main.py
    ├── test_listener.py
    ├── test_retriever.py
    ├── test_composer.py
    ├── test_brain.py
    ├── test_speaker.py
    ├── test_status_led.py
    ├── test_config.py
    ├── test_http_client.py
    └── test_audio_utils.py
```

## 必要依存ライブラリ

### 開発・テスト用
```bash
sudo apt install python3-pytest python3-pytest-mock python3-numpy python3-requests
```

### 本番ロジック用
```bash
pip install faster-whisper piper-tts-plus requests
```

## セットアップ

1. **環境変数の設定**
   ```bash
   cp .env.example .env
   # .env を編集して環境に合わせて設定
   ```

2. **外部サービスの起動（Docker）**
   ```bash
   # SearXNG
   docker run -d -p 8080:8080 --name searxng searxng/searxng

   # Ollama
   docker run -d -p 11434:11434 --name ollama ollama/ollama
   ```

3. **Piper 音声モデルの配置**
   ```bash
   mkdir -p models
   # 日本語 ONNX モデル（例: tsukuyomi.onnx）と config.json を models/ に配置
   ```

## 実行

```bash
python3 src/main.py
```

## テスト

```bash
python3 -m pytest tests/ -v
```
