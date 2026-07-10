# AGENTS.md

AI エージェントがこのリポジトリでコードを変更するときに守る**開発ルール**。
プロジェクトの要件・アーキテクチャは `project_context.md`、セットアップ・実行手順は `README.md` を参照すること。

## 作業前

1. 変更内容が要件に合うか `project_context.md` で確認する
2. セットアップや実行方法が必要なら `README.md` を参照する
3. 未実装の拡張は `future_extensions.md` を確認し、**明示的な依頼がない限り着手しない**

## 設計上の制約（コード変更時）

以下は `project_context.md` の要件を、実装・修正時に守るためのルールである。

- **生成前 RAG** の処理順序を崩さない（検索 → プロンプト構成 → 生成）
- `composer.py` のプロンプトは、検索結果を事実として扱う指示を維持する
- 回答は **日本語のみ**（アルファベットを含めない）。Piper TTS の安定性のため
- TTS は `piper-tts-plus` のみ使用する。本家 Piper（espeak-ng 依存）は使わない
- `status_led.py` は非 Pi 環境で自動無効化される現行挙動を維持する
- 環境変数は **`src/config.py` に集約** する。各モジュールで `os.getenv()` を直接書かない
- `main.py` のエラー方針に合わせる:
  - 検索失敗は **非致命**（空コンテキストで続行）
  - 録音・認識・生成・発話の失敗は **致命**（適切な例外を送出）

## コーディング規約

### スタイル

- Python 3。モジュール先頭に `#!/usr/bin/env python3` と docstring
- クラスベースのコンポーネント設計（`Listener`, `Retriever`, `Composer`, `Brain`, `Speaker`, `StatusLED`）
- ログは `logging.getLogger(__name__)` を使用
- 初期化ログは `audio_utils.log_init` / `log_ready` を使う

### エラーハンドリング

- ドメイン例外は `src/exceptions.py` に定義する
  - `ListenerError`, `SearchError`, `GenerationError`, `SpeakerError`
- HTTP 通信は `src/http_client.py` の `http_get_json` / `http_post_json` を使い、呼び出し側の `error_class` で例外を分ける
- URL は `config.validate_url()` で検証する
- 低レベル例外は `raise XxxError(...) from e` でラップする

### 変更時の原則

- **最小スコープ**: 依頼された問題だけを直す。無関係なリファクタリングをしない
- **既存パターンを踏襲**: 周辺コードの命名・構造・テストスタイルに合わせる
- **過剰な抽象化を避ける**: 1〜2 行のヘルパーや不要なエラーハンドリングを増やさない
- **コメント**: 自明な処理には書かない。非自明なビジネスロジック・ハードウェア制約のみ

## テスト

- コード変更後は `python3 -m pytest tests/ -v` を実行し、結果を報告する
- 新規 `src/*.py` には対応する `tests/test_*.py` を追加する（ミラーテスト）
- **外部 API（SearXNG / Ollama）へ実リクエストを送らない**
- `pytest-mock` または `unittest.mock` でモック化する
- 外部依存（`faster_whisper`, `piper`, `gpiozero` 等）は import 前にモックする既存パターンに従う

## コミット

- `.env`, `models/`, `*.wav`, `*.log` はコミットしない
- 設定変更時は `.env.example` を更新する
- セットアップ手順に影響がある変更は `README.md` を更新する
- 要件に影響がある変更は `project_context.md` を更新する

## やってはいけないこと

- 本家 Piper（espeak-ng）への差し替え
- テストから実 SearXNG / Ollama への接続
- `config.py` を経由しない環境変数の直書き
- 英語混在プロンプトへの変更
- 生成後 RAG への設計変更
- `future_extensions.md` の機能を依頼なく実装する
