# AGENTS.md

AIエージェント（Devin など）がこのリポジトリで作業する際に、全セッションで必ず守るルール。

## プロジェクト概要

ローカルRAG構成のAIスマートスピーカー（Raspberry Pi 5）。ハルシネーションを防ぐため
**生成前RAG**（Web検索の結果を事前にプロンプトへ含める）を採用している。処理パイプラインは
`main.py` が統括し、`listener → retriever → composer → brain → speaker` の順に流れる。
`status_led.py` が各状態を GPIO 接続の LED で視覚化する。

詳細な要件は `project_context.md`、将来の拡張バックログは `future_extensions.md` を参照。

## セットアップ

```bash
# 開発・テスト用
pip install pytest pytest-mock numpy requests

# 本番ロジック用（Piper は必ず fork 版）
pip install faster-whisper piper-tts-plus requests

# 環境変数
cp .env.example .env   # 必要に応じて編集
```

## テスト（必須）

コードを変更したら、コミット前に**必ず**テストを実行してパスさせること。

```bash
python3 -m pytest tests/ -v
```

- テストは `src/` のレイアウトをミラーする（`src/foo.py` → `tests/test_foo.py`）。
- 新しいモジュールを追加したら、対応する `tests/test_*.py` も必ず追加する。
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

## コンポーネント別の注意点

- **speaker.py**: 本家 Piper（espeak-ng 依存）は日本語アクセント解析が未対応。
  必ず日本語特化 fork の `piper-tts-plus` を使うこと。音声モデルは日本語 ONNX + JSON 設定を使用。
- **retriever.py / brain.py**: 外部 API 通信部。テストでは必ずモック化する。
- **status_led.py**: `gpiozero` は Raspberry Pi 上でのみ動作する。非 Pi 環境や GPIO 未接続時は
  自動的に無効化される設計を壊さないこと（import 失敗・実行時失敗を握りつぶして継続する）。

## コミット・PR

- 秘密情報（`.env`, 音声モデル `models/`, `*.wav`, `*.log`）はコミットしない（`.gitignore` 済み）。
- コミット前に必ず `python3 -m pytest tests/ -v` が全パスすることを確認する。
- 1 PR は 1 つの目的に絞る。
