# AGENTS.md

AIエージェント（Devin など）がこのリポジトリで作業する際に、全セッションで必ず守る**開発ルール**。

- プロジェクトの内容・要件は `project_context.md`（将来の拡張は `future_extensions.md`）を参照。
- セットアップ・動かし方は `README.md` を参照。
- 本ファイルは開発ルールのみを扱う。

## 設計上の制約

`project_context.md` の要件を、実装・修正時に守るためのルール。

- **生成前RAG** の基本順序を崩さない（検索要否判定・クエリ書き換え → 検索 → Reranking → 本文取得 → パッセージ再ランク → プロンプト構成 → 生成）。
  - **最終回答の生成は、検索とコンテキスト構成の後に1回だけ**行う。生成した回答を LLM で
    再度検証・修正・推敲する多段生成（生成後検証）は導入しない。発話前の単位・略語の
    機械的な日本語読み化（`speech_normalize.py`）は LLM 推敲ではないので、この禁止の対象外。
  - 最終回答の生成より**前**の補助的な LLM 呼び出し（クエリ書き換え・検索要否判定・
    エビデンス抽出など）は追加してよい。これらは順序違反とみなさない。
- **最終的に発話する回答**は日本語のみ（アルファベットを含めない）。TTS（`piper-tts-plus`）の
  安定性のため、日本語限定の指示はプロンプトまたは system メッセージのいずれかで維持し、
  残った単位・略語は発話前に日本語読みへ正規化する（置き場所は `composer.py` に限定しない）。
- この日本語限定の制約は**発話される出力にのみ適用**する。クエリ書き換え・検索要否判定など、
  発話されない補助的な LLM 呼び出しの出力は対象外とし、英字を含む検索クエリを返してよい。
  補助呼び出しの結果をそのまま発話してはならない。
- `main.py` のエラー方針に合わせる:
  - 検索失敗（`SearchError`）は**非致命**。空コンテキストで処理を続行する。
  - 補助的な LLM 呼び出し（クエリ書き換え・検索要否判定・エビデンス抽出など）と
    検索結果ページの本文取得の失敗も**非致命**。その段を通さなかった場合の結果へ degrade して
    処理を続行し、`logger.warning` を残す（Reranking 失敗時のフォールバックと同じ考え方）。
  - **最終回答の生成**（`GenerationError`）・録音・認識（`ListenerError`）・
    発話（`SpeakerError`）の失敗は**致命**。
- `future_extensions.md` の機能は、**明示的な依頼がない限り着手しない**。
  実装したら同じ PR で `project_context.md` に記載を移し、`future_extensions.md` からは削除する。
- **ルールの先回り緩和はしない**。将来やるかもしれない変更のために制約を緩めず、実際に着手する
  時点で、その変更に必要な分だけ改訂する。
- 本ファイルと `project_context.md` で重複する記述（処理順序・エラー方針・依存ライブラリなど）は
  **必ず同時に更新**する。片方だけ直して、矛盾した指示を残さない。

## テスト

- コードを変更したら、コミット前に**必ず** `python3 -m pytest tests/ -v` を実行し、全パスさせる。
- テストは `src/` のレイアウトをミラーする（`src/foo.py` → `tests/test_foo.py`）。
  新しいモジュールを追加したら、対応する `tests/test_*.py` も必ず追加する。
  この規約は `src/` 配下のみに適用し、`scripts/` は対象外。
- **`tests/` 配下では**、外部サービス（SearXNG / Ollama）やハードウェア（マイク・GPIO）に
  **絶対に実アクセスしない**。`pytest-mock`（`mocker` フィクスチャ）でモック化すること。
  外部サービスが起動していない環境でも全テストが通る状態を保つ。
- 外部依存ライブラリ（`faster_whisper`, `piper`, `gpiozero` 等）は、import 前にモックする既存パターンに従う。
- 共通のテスト用データは `tests/conftest.py` のフィクスチャ
  （`mock_audio_array`, `mock_search_results`, `mock_transcribed_text`）を再利用する。
- ルートの `conftest.py` が `src/` を `sys.path` に追加しているため、
  テストは `from src.X import ...` でモジュールを import できる。

## スクリプト（scripts/）

pytest で自動実行しない、手動実行のスクリプトを置く。テストの外部アクセス禁止と、
評価やハードウェア確認の必要性を両立させるためのルール。

- `tests/` には置かない。`python3 -m pytest tests/ -v` の対象外とする。
- 設定値は `src/config.py` 経由で読む。`scripts/` でも `os.getenv()` を直接呼ばない。
- 実行者に向けた対話メッセージは `print()` を使ってよい（`src/` 配下の `print()` 禁止は適用しない）。
- **評価・ベンチマーク用**: 実サービス（SearXNG / Ollama）への接続を許可する。ただしハードウェア
  （マイク・スピーカー・GPIO）には依存させず、テキスト入出力だけでパイプラインを実行できるようにする。
- 回答精度に影響する変更をしたら、`scripts/eval.py` で改善前後を比較する。
- **ハードウェア確認用**: GPIO などの実ハードウェアに直接アクセスしてよい。Raspberry Pi 上で
  手動実行する前提とする。ピン番号などの設定値は `src/config.py` から読み、スクリプト内に
  直接書かない（設定が二重管理になり、実装と食い違う原因になる）。

## コーディング規約

- Python 3。標準ライブラリと既存の依存を優先し、新規依存は最小限にする。
- **依存の正は `pyproject.toml`**。依存を追加・変更したら、`pyproject.toml` と `README.md` の
  セットアップ手順、`project_context.md` の依存ライブラリ一覧をあわせて更新する。
- 新規依存を追加する場合は、PR 説明に「既存の依存で代替できない理由」と
  「Raspberry Pi（ARM64）での動作可否・メモリ使用量・起動時間への影響」を書く。
- 各モジュールは先頭に `#!/usr/bin/env python3` とモジュール docstring を置く既存スタイルに合わせる。
- **設定値は必ず `src/config.py` 経由**で読む。モジュール内で `os.getenv()` を直接呼ばない
  （設定の重複を避けるため一元管理している）。
- **HTTP 通信は `src/http_client.py` に集約する**（既存は `http_get_json` / `http_get_text` / `http_post_json`。
  JSON 以外を扱うなど既存関数で対応できない場合は、同じファイルに関数を追加する）。
  `requests` を各モジュールで直接呼び出して重複したエラーハンドリングを書かない。
  URL は `config.validate_url()`（`http` / `https` のみ許可）で検証する。
- **例外は `src/exceptions.py`** のドメイン固有例外を使う
  （`ListenerError`, `SearchError`, `GenerationError`, `SpeakerError`。基底は `VoiceAssistantError`）。
  裸の `Exception` を投げない。低レベル例外は `raise XxxError(...) from e` でラップする。
- ロギングは `logging.getLogger(__name__)` を使う。**`src/` 配下では `print()` を使わない**
  （`scripts/` の手動実行スクリプトのみ例外。`tests/test_logging_policy.py` が検証する）。
  初期化・準備完了のログは `audio_utils.log_init` / `log_ready` を使う。
- 変更は最小限・対象を絞る。無関係なファイルやテストを書き換えない。

## コンポーネント別の実装ルール

- **speaker.py**: 本家 Piper（espeak-ng 依存）は使わない。必ず日本語特化 fork の
  `piper-tts-plus` を使うこと（音声モデルは日本語 ONNX + JSON 設定）。
- **retriever.py / brain.py**: 外部 API 通信部。テストでは必ずモック化する。
- **retriever.py**: 結果 Rerank のあと、上位 URL を並列取得して `trafilatura` で本文抽出し、
  パッセージに分割して同じクロスエンコーダで二段 Rerank する。本文取得・抽出・二段 Rerank
  の失敗は非致命で、その URL のスニペットまたは取得済み本文へ degrade する。
  `trafilatura` は使う直前の動的 import。ページ取得は `http_get_text` を使い、
  `retriever.py` から `requests` を直接呼ばない。
- **brain.py**: `/api/chat` に `messages` / `keep_alive` / `options`（`num_ctx` / `num_predict` / `temperature` / `repeat_penalty`）を渡す。日本語限定の指示は `OLLAMA_SYSTEM_PROMPT`（system メッセージ）で維持する。補助呼び出しは `generate_auxiliary()`（`OLLAMA_AUX_NUM_PREDICT` / `OLLAMA_AUX_TEMPERATURE`）を使い、spoken system prompt は付けない。設定は `src/config.py` 経由。`prompt[:10000]` は使わず、検索コンテキストは `CONTEXT_CHAR_BUDGET`、履歴はトークン予算で削る。
- **composer.py**: 本番は `compose_messages()`。日本語限定を維持する。文数は質問タイプに合わせる（事実は1〜3文、説明は3〜5文、結論先出し）。検索結果は関係するものだけを事実とし、無関係なら使わず、使う場合は推測で補わない。発話に「検索結果」とは言わない。`compose_prompt()` は切り戻し用に残す。
- **query_prep.py**: 検索要否とクエリ書き換えを1回の補助LLM呼び出しで行う。失敗は非致命で元の質問へ degrade する。結果は発話しない。
- **text_turn.py**: `main.py` と `scripts/eval.py` が同じテキストターンを使う。生成後に `normalize_for_speech()` する。
- **status_led.py**: 非 Pi 環境や GPIO 未接続時に `gpiozero` を自動無効化する設計を壊さない
  （import 失敗・実行時失敗を握りつぶして処理を継続する）。
- **push_to_talk.py**: 同じく `gpiozero` が使えない環境では自動無効化し、`RECORD_SECONDS` の
  固定秒数録音にフォールバックする設計を壊さない。

## コミット・PR

- 秘密情報（`.env`, 音声モデル `models/`, `*.wav`, `*.log`）はコミットしない（`.gitignore` 済み）。
- 設定項目を追加・変更したら、**`src/config.py` / `.env.example` / `README.md` の3点を同期**する。
  既定値は `src/config.py` と `.env.example` で必ず一致させる。
  新しい設定項目には、`tests/test_config.py` に既定値の検証を追加する。
- 既定値は**`.env` を置かなくても実機（Raspberry Pi の既定配線・既定のモデル配置）で
  そのまま動く値**にする。プレースホルダや開発中の暫定値を残さない。
- コミット前に必ず `python3 -m pytest tests/ -v` が全パスすることを確認する。
- 1 PR は 1 つの目的に絞る。
- `main` ブランチへの直接 push は禁止。変更は必ず feature ブランチを作り、プルリクエストを経由して統合する。
