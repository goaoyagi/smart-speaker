# AIスマートスピーカー開発要件（ラズパイ5）

## ドキュメントの役割

| ファイル | 内容 |
|----------|------|
| `project_context.md` | 本ドキュメント（要件・アーキテクチャ・設計方針） |
| `README.md` | セットアップ・実行手順 |
| `AGENTS.md` | AI エージェント向け開発ルール |
| `future_extensions.md` | 未実装機能のバックログ |

## 1. プロジェクト概要

- **目的**: ハルシネーションを抑えた、事実ベースの音声アシスタント
- **実行環境**: Raspberry Pi 5（開発・CI は非 Pi 環境でも可）
- **言語**: ユーザー向け応答・プロンプトは **日本語のみ**（Piper TTS の安定性のため）

## 2. システムアーキテクチャ（ローカルRAG構成）

ハルシネーション（嘘）を徹底的に防ぐため、以下の順序で処理を行う。生成後の検証ではなく、**「生成前RAG（あらかじめカンニングペーパーを渡す）」を徹底**すること。

`src/main.py` の `VoiceAssistant` が全体を統括する。

1. **[耳] listener.py**: `faster-whisper` でユーザーの音声をテキスト化
2. **[検索] retriever.py**: 質問をトリガーに、ローカルの SearXNG で Web 検索を実行
3. **[構成] composer.py**: 検索結果（事実ソース）と質問をプロンプトに編成
4. **[脳] brain.py**: プロンプトを Ollama（Qwen2.5:3b）に投入し、事実に基づく回答を生成
5. **[口] speaker.py**: `piper-tts-plus` で音声合成して発話
6. **[視覚] status_led.py**: GPIO 接続の LED で待機・聞き取り・検索・思考・発話・エラーを表示

### 共通モジュール

- **config.py**: 環境変数の一元管理（各モジュールで `os.getenv()` を直接呼ばない）
- **exceptions.py**: パイプライン固有の例外（`ListenerError`, `SearchError`, `GenerationError`, `SpeakerError`）
- **http_client.py**: SearXNG / Ollama への HTTP 通信の共通化
- **audio_utils.py**: 音声ファイル操作・初期化ログの共通処理

### エラー処理方針

- 検索失敗は **非致命**（空コンテキストで続行）
- 録音・認識・生成・発話の失敗は **致命**（適切な例外を送出）

## 3. 外部サービス

| サービス | 用途 | デフォルト |
|----------|------|------------|
| SearXNG | Web 検索 | `http://localhost:8080` |
| Ollama | LLM 生成 | `http://localhost:11434/api/generate`（モデル: `qwen2.5:3b`） |

音声認識: Whisper `small`（CPU / int8）  
音声合成: Piper 日本語 ONNX モデル（`models/` 配下、Git 管理外）

環境変数の詳細は `.env.example` を参照。

## 4. 必要依存ライブラリ

### 開発・テスト用

- **pytest**: テストフレームワーク
- **pytest-mock**: 外部 API やハードウェアのモック化
- **numpy**: 音声データ処理（テスト含む）

### 本番ロジック用

- **faster-whisper**: 音声認識
- **piper-tts-plus**: 日本語特化の音声合成（OpenJTalk 内蔵版）
- **requests**: SearXNG / Ollama との通信
- **gpiozero**: ステータス LED 制御（Raspberry Pi 上のみ）

## 5. 各コンポーネントの実装注意点

### listener.py（音声認識）

- 音声入出力は `arecord` / `aplay`（ALSA）を使用
- マイク・サンプルレート等は `config.py` 経由で設定

### retriever.py / brain.py（外部通信）

- HTTP 通信は `http_client.py` を使用する
- URL は `config.validate_url()` で検証する

### composer.py（プロンプト構成）

- 検索結果を「絶対に事実」として扱う指示を含める
- 回答は日本語のみ（アルファベットを含めない）

### speaker.py（音声合成）

- 本家 Piper（espeak-ng 依存）は日本語のアクセント解析が未対応のため使用しない
- 必ず `piper-tts-plus` を使用する
- 音声モデルは Hugging Face 等から日本語 ONNX モデルと JSON 設定ファイルを取得して利用する

### status_led.py（視覚フィードバック）

- `gpiozero` は Raspberry Pi 上でのみ動作する
- 非 Pi 環境や GPIO 未接続時は自動的に無効化する

## 6. ディレクトリ構成（src レイアウト・ミラーテスト）

```text
smart-speaker/
├── project_context.md      # 本ドキュメント
├── README.md               # セットアップ・実行手順
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
│   ├── listener.py
│   ├── retriever.py
│   ├── composer.py
│   ├── brain.py
│   ├── speaker.py
│   └── status_led.py
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

テストは `src/` のレイアウトをミラーする（`src/foo.py` → `tests/test_foo.py`）。
