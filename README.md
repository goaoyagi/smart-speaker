# AI Smart Speaker (Raspberry Pi 5)

ローカル RAG 構成の AI スマートスピーカー。ハルシネーションを防ぐため、生成前 RAG（検索結果を事前にプロンプトに含める）を採用。

## ドキュメント

| ファイル | 内容 |
|----------|------|
| `README.md` | 本ファイル（セットアップ・実行手順） |
| `project_context.md` | プロジェクト要件・アーキテクチャ |
| `AGENTS.md` | AI エージェント向け開発ルール |
| `future_extensions.md` | 未実装機能のバックログ |

## アーキテクチャ

1. **[耳] listener.py**: Whisper でユーザーの音声をテキスト化
2. **[検索] retriever.py**: 質問をトリガーに、ローカルの SearXNG で Web 検索を実行
3. **[構成] composer.py**: 検索結果（事実ソース）と質問をプロンプトに編成
4. **[脳] brain.py**: プロンプトを Ollama（Qwen2.5:3b）に投入し、事実に基づく回答を生成
5. **[口] speaker.py**: `piper-tts-plus` で音声合成して発話
6. **[視覚] status_led.py**: GPIO 接続の LED で、待機・聞き取り・検索・思考・発話・エラーを表示

詳細な要件は `project_context.md` を参照。

## 必要依存ライブラリ

### 開発・テスト用

```bash
sudo apt install python3-pytest python3-pytest-mock python3-numpy python3-requests
```

### 本番ロジック用

```bash
pip install faster-whisper piper-tts-plus requests
```

Raspberry Pi 上でステータス LED を使う場合:

```bash
pip install gpiozero
```

## セットアップ

### 1. 環境変数の設定

```bash
cp .env.example .env
# .env ファイルを編集して環境に合わせて設定
```

主な設定項目:

| 変数 | 説明 | デフォルト |
|------|------|------------|
| `MIC_DEVICE` | マイクデバイス（ALSA） | `hw:0,0` |
| `SPEAKER_DEVICE` | スピーカーデバイス（ALSA） | `plughw:0,0` |
| `WHISPER_MODEL_SIZE` | Whisper モデルサイズ | `small` |
| `PIPER_MODEL_PATH` | Piper ONNX モデルパス | `./models/tsukuyomi.onnx` |
| `SEARXNG_URL` | SearXNG の URL | `http://localhost:8080` |
| `OLLAMA_API_URL` | Ollama API の URL | `http://localhost:11434/api/generate` |
| `OLLAMA_MODEL` | Ollama モデル名 | `qwen2.5:3b` |
| `STATUS_LED_ENABLED` | LED 有効化 | `true` |
| `STATUS_LED_PIN` | GPIO ピン番号（BCM） | `23` |

全項目は `.env.example` を参照。

### 2. Docker コンテナの起動

```bash
# SearXNG
docker run -d -p 8080:8080 --name searxng searxng/searxng

# Ollama（ローカル実行の場合は不要）
docker run -d -p 11434:11434 --name ollama ollama/ollama
```

### 3. 音声モデルの配置

```bash
mkdir -p models
# Piper TTS の ONNX モデルと config.json を models/ に配置
```

## 実行

```bash
# 仮想環境を使用
source venv/bin/activate
python3 src/main.py

# またはシステム Python を使用
python3 src/main.py
```

## テスト

```bash
python3 -m pytest tests/ -v
```

外部 API（SearXNG / Ollama）やハードウェアには接続せず、モックで実行される。

## ディレクトリ構成

```
smart-speaker/
├── project_context.md      # プロジェクト要件
├── README.md               # 本ファイル
├── AGENTS.md               # AI エージェント向け開発ルール
├── future_extensions.md    # 将来拡張バックログ
├── .env.example            # 環境変数テンプレート
├── conftest.py             # テスト用 sys.path 設定
├── src/
│   ├── main.py             # オーケストレーター
│   ├── config.py           # 環境変数の一元管理
│   ├── exceptions.py       # ドメイン例外
│   ├── http_client.py      # HTTP 共通処理
│   ├── audio_utils.py      # 音声・ログ共通ユーティリティ
│   ├── listener.py         # Whisper 音声認識
│   ├── retriever.py        # SearXNG Web 検索
│   ├── composer.py         # RAG プロンプト構成
│   ├── brain.py            # Ollama AI 生成
│   ├── speaker.py          # Piper-Plus TTS
│   └── status_led.py       # GPIO ステータス LED
└── tests/
    ├── conftest.py         # pytest 共通フィクスチャ
    ├── test_main.py
    ├── test_config.py
    ├── test_http_client.py
    ├── test_audio_utils.py
    ├── test_listener.py
    ├── test_retriever.py
    ├── test_composer.py
    ├── test_brain.py
    ├── test_speaker.py
    └── test_status_led.py
```

## 注意点

- **speaker.py**: 本家 Piper（espeak-ng 依存）は日本語のアクセント解析が未対応のため、必ず `piper-tts-plus` を使用すること
- **status_led.py**: `gpiozero` は Raspberry Pi 上でのみ動作するため、非 Pi 環境では自動的に無効化される
- **models/**: 音声モデルは Git 管理外（`.gitignore` 済み）
