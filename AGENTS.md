# AGENTS.md — AI Agent Instructions for smart-speaker

このファイルは、GitHub Copilot などの AI エージェントがこのリポジトリで作業する際の規約・制約・手順を定めたものです。

---

## プロジェクト概要

ラズパイ 5 上で動作する**ローカル RAG 構成**の日本語 AI スマートスピーカー。  
ハルシネーションを防ぐため、**生成前 RAG**（検索結果をあらかじめプロンプトに含める）を採用している。

---

## ディレクトリ構成

```
smart-speaker/
├── AGENTS.md               # 本ファイル
├── README.md
├── project_context.md      # プロジェクト要件の原本
├── future_extensions.md    # 将来拡張バックログ
├── .env.example            # 環境変数テンプレート
├── conftest.py             # ルート conftest（sys.path 設定）
├── src/
│   ├── __init__.py
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
    ├── __init__.py
    ├── conftest.py          # 共通フィクスチャ（mock_audio_array 等）
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

---

## アーキテクチャ（パイプライン）

```
マイク入力
   │
   ▼
listener.py   — arecord で録音 → faster-whisper で日本語テキスト化
   │
   ▼
retriever.py  — SearXNG（localhost:8080）で Web 検索、上位 5 件取得
   │
   ▼
composer.py   — 検索結果 + 質問をプロンプトに編成（日本語のみ出力を強制）
   │
   ▼
brain.py      — Ollama（localhost:11434、Qwen2.5:3b）でレスポンス生成
   │
   ▼
speaker.py    — piper-tts-plus で音声合成 → aplay で再生
```

`status_led.py` は各ステップに連動して GPIO LED 状態（IDLE / LISTENING / SEARCHING / THINKING / SPEAKING / ERROR）を更新する。

---

## 環境変数

`.env.example` をコピーして `.env` を作成する。設定値はすべて `src/config.py` で `os.getenv()` により読み込まれ、各モジュールはここから `import` する（モジュール内での直接 `os.getenv()` 呼び出しは禁止）。

| 変数名 | デフォルト | 説明 |
|---|---|---|
| `MIC_DEVICE` | `hw:0,0` | ALSA マイクデバイス |
| `SPEAKER_DEVICE` | `plughw:0,0` | ALSA スピーカーデバイス |
| `SAMPLE_RATE` | `16000` | サンプリングレート (Hz) |
| `CHANNELS` | `1` | チャンネル数 |
| `RECORD_SECONDS` | `10` | 録音秒数 |
| `WHISPER_MODEL_SIZE` | `small` | Whisper モデルサイズ |
| `PIPER_MODEL_PATH` | `./models/tsukuyomi.onnx` | Piper ONNX モデルパス |
| `PIPER_CONFIG_PATH` | `./models/config.json` | Piper 設定 JSON パス |
| `STATUS_LED_ENABLED` | `true` | LED 有効化フラグ |
| `STATUS_LED_PIN` | `23` | GPIO ピン番号 |
| `SEARXNG_URL` | `http://localhost:8080` | SearXNG エンドポイント |
| `OLLAMA_API_URL` | `http://localhost:11434/api/generate` | Ollama API URL |
| `OLLAMA_MODEL` | `qwen2.5:3b` | 使用モデル名 |
| `DEBUG_AUDIO_DIR` | （空） | デバッグ用音声保存ディレクトリ |

---

## テスト

```bash
python3 -m pytest tests/ -v
```

### テスト作成の必須ルール

- **外部サービスへのリクエストを絶対に送らないこと。**  
  SearXNG・Ollama など、外部 API / ローカルサーバーへの通信はすべて `pytest-mock` でモック化する。
- `tests/conftest.py` に定義済みのフィクスチャ（`mock_audio_array`、`mock_search_results`、`mock_transcribed_text`）を積極的に再利用する。
- ハードウェア依存処理（`arecord`、`aplay`、`gpiozero`）は `mocker.patch` で差し替える。
- 新モジュール `src/foo.py` を追加した場合は、対応する `tests/test_foo.py` を必ず作成する（ミラーテスト構成）。

---

## コーディング規約

### 例外処理

`src/exceptions.py` に定義されたカスタム例外を使用すること。

```
VoiceAssistantError          # 基底クラス
├── ListenerError            # 録音・文字起こし失敗
├── SearchError              # SearXNG 通信失敗
├── GenerationError          # Ollama 生成失敗
└── SpeakerError             # TTS 合成・再生失敗
```

- パイプライン内では各ステップで適切な例外をそのまま raise する。
- `SearchError` は致命的ではないため `main.py` でキャッチして継続する（検索結果なしでプロンプト続行）。
- `ListenerError`・`GenerationError`・`SpeakerError` は致命的エラーとして扱う。

### TTS（音声合成）

- **必ず `piper-tts-plus` を使用すること。** 本家 `piper`（espeak-ng 依存）は日本語アクセント解析が未対応のため使用禁止。
- モデルは日本語 ONNX モデル（JVS データセット等）を `models/` に配置する。

### Status LED

- `gpiozero` は Raspberry Pi 上でのみ動作する。非 Pi 環境では `StatusLED.__init__` が例外をキャッチして `_enabled = False` に設定し、以降の `set_state()` 呼び出しは無操作となる。
- **LED の初期化失敗はパイプライン全体を止めてはならない。**

### 出力言語

- `composer.py` が生成するプロンプトは日本語のみの出力を LLM に強制する（アルファベット禁止指示を含む）。この制約はプロンプトテンプレートを変更する際も維持すること。

### HTTP 通信

- `src/http_client.py` の `http_get_json` / `http_post_json` を通じて行う。モジュール内で直接 `requests` を呼ばないこと。
- URL は `config.py` の `validate_url()` で事前に検証する（`http` / `https` のみ許可）。

### ログ

- 各モジュール冒頭で `logger = logging.getLogger(__name__)` を宣言し、`logger.info` / `logger.warning` / `logger.error` を使う。
- 初期化・準備完了のログには `audio_utils.log_init()` / `audio_utils.log_ready()` を使う。

---

## セットアップ手順

```bash
# 1. 依存インストール（開発・テスト用）
sudo apt install python3-pytest python3-pytest-mock python3-numpy python3-requests

# 2. 依存インストール（本番用）
pip install faster-whisper piper-tts-plus requests

# 3. 環境変数設定
cp .env.example .env
# .env を編集して環境に合わせて設定

# 4. 外部サービス起動（Docker）
docker run -d -p 8080:8080 --name searxng searxng/searxng
docker run -d -p 11434:11434 --name ollama ollama/ollama

# 5. Piper モデル配置
mkdir -p models
# 日本語 ONNX モデルと config.json を models/ に配置

# 6. 実行
python3 src/main.py
```

---

## 将来拡張（`future_extensions.md` 参照）

実装済みではないが、設計時に考慮すること：

- **ウェイクワード起動**（OpenWakeWord / Porcupine）
- **ローカルドキュメント検索**（ChromaDB / FAISS によるプライベート RAG）
- **会話コンテキスト保持**（`conversation_history.py` + sliding window memory）

これらを実装する際は、既存の `retriever.py`・`composer.py`・`main.py` の責務を尊重して拡張すること。
