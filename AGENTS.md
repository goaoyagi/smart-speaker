# AGENTS.md — AI エージェント向け開発ルール

このファイルは GitHub Copilot などの AI エージェントがこのリポジトリで作業する際の**開発ルール**を定めます。  
プロジェクトの概要・要件は `project_context.md`、セットアップ・実行方法は `README.md` を参照してください。

---

## コーディング規約

### 設定値（環境変数）

- 環境変数はすべて `src/config.py` で一元管理する。
- 各モジュールは `from .config import XXX` で値を取得し、**モジュール内で直接 `os.getenv()` を呼ばないこと。**

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

## テスト規約

- **外部サービスへのリクエストを絶対に送らないこと。**  
  SearXNG・Ollama など、外部 API / ローカルサーバーへの通信はすべて `pytest-mock` でモック化する。
- `tests/conftest.py` に定義済みのフィクスチャ（`mock_audio_array`、`mock_search_results`、`mock_transcribed_text`）を積極的に再利用する。
- ハードウェア依存処理（`arecord`、`aplay`、`gpiozero`）は `mocker.patch` で差し替える。
- 新モジュール `src/foo.py` を追加した場合は、対応する `tests/test_foo.py` を必ず作成する（ミラーテスト構成）。

---

## 将来拡張実装時の注意

`future_extensions.md` に記載された機能（ウェイクワード起動・プライベート RAG・会話コンテキスト保持）を実装する際は、既存の `retriever.py`・`composer.py`・`main.py` の責務を尊重して拡張すること。
