# AGENTS.md

AIエージェント（Devin など）がこのリポジトリで作業する際に、全セッションで必ず守る**開発ルール**。

- プロジェクトの内容・要件は `project_context.md`（将来の拡張は `future_extensions.md`）を参照。
- セットアップ・動かし方は `README.md` を参照。
- 本ファイルは開発ルールのみを扱う。

## テスト

- コードを変更したら、コミット前に**必ず** `python3 -m pytest tests/ -v` を実行し、全パスさせる。
- テストは `src/` のレイアウトをミラーする（`src/foo.py` → `tests/test_foo.py`）。
  新しいモジュールを追加したら、対応する `tests/test_*.py` も必ず追加する。
- 外部サービス（SearXNG / Ollama）やハードウェア（マイク・GPIO）には**絶対に実アクセスしない**。
  `pytest-mock`（`mocker` フィクスチャ）でモック化すること。
- 共通のテスト用データは `tests/conftest.py` のフィクスチャ
  （`mock_audio_array`, `mock_search_results`, `mock_transcribed_text`）を再利用する。
- ルートの `conftest.py` が `src/` を `sys.path` に追加しているため、
  テストは `from src.X import ...` でモジュールを import できる。

## コーディング規約

- Python 3、標準ライブラリと既存の依存（`requests`, `numpy`）を優先。新規依存は最小限に。
- 各モジュールは先頭に `#!/usr/bin/env python3` とモジュール docstring を置く既存スタイルに合わせる。
- **設定値は必ず `src/config.py` 経由**で読む。モジュール内で `os.getenv()` を直接呼ばない
  （設定の重複を避けるため一元管理している）。
- **HTTP 通信は `src/http_client.py`** の `http_get_json` / `http_post_json` を使う。
  `requests` を各モジュールで直接呼び出して重複したエラーハンドリングを書かない。
- **例外は `src/exceptions.py`** のドメイン固有例外を使う
  （`ListenerError`, `SearchError`, `GenerationError`, `SpeakerError`。基底は `VoiceAssistantError`）。
  裸の `Exception` を投げない。低レベル例外は `raise XxxError(...) from e` でラップする。
- ロギングは `logging.getLogger(__name__)` を使い、`print()` でのデバッグ出力を残さない。
- 変更は最小限・対象を絞る。無関係なファイルやテストを書き換えない。

## コンポーネント別の実装ルール

- **speaker.py**: 本家 Piper（espeak-ng 依存）は使わない。必ず日本語特化 fork の
  `piper-tts-plus` を使うこと（音声モデルは日本語 ONNX + JSON 設定）。
- **retriever.py / brain.py**: 外部 API 通信部。テストでは必ずモック化する。
- **status_led.py**: 非 Pi 環境や GPIO 未接続時に `gpiozero` を自動無効化する設計を壊さない
  （import 失敗・実行時失敗を握りつぶして処理を継続する）。

## コミット・PR

- 秘密情報（`.env`, 音声モデル `models/`, `*.wav`, `*.log`）はコミットしない（`.gitignore` 済み）。
- コミット前に必ず `python3 -m pytest tests/ -v` が全パスすることを確認する。
- 1 PR は 1 つの目的に絞る。
