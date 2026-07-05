# AGENTS.md — smart-speaker

Raspberry Pi 5 上で動作するローカル RAG 構成の AI スマートスピーカー。  
このファイルは Cursor Agent が **全セッションで自動参照** するリポジトリ共通ルールです。

## プロジェクト概要

- **目的**: ハルシネーションを抑えた、事実ベースの音声アシスタント
- **言語**: ユーザー向け応答・プロンプトは **日本語のみ**（Piper TTS の安定性のため）
- **実行環境**: Raspberry Pi 5（開発・CI は非 Pi 環境でも可）
- **詳細要件**: `project_context.md` を正とする
- **将来拡張**: `future_extensions.md` を参照（未実装機能のバックログ）

## アーキテクチャ（変更禁止の基本フロー）

処理順序は **生成前 RAG** を徹底する。生成後検証ではなく、検索結果を先にプロンプトへ含める。

```
[耳] listener.py   → Whisper で音声認識
[検索] retriever.py → SearXNG で Web 検索
[構成] composer.py  → 検索結果 + 質問をプロンプト化
[脳] brain.py       → Ollama (Qwen2.5:3b) で回答生成
[口] speaker.py     → piper-tts-plus で音声合成・再生
[視覚] status_led.py → GPIO LED で状態表示
```

オーケストレーションは `src/main.py` の `VoiceAssistant` が担当する。

## ディレクトリ構成

```
smart-speaker/
├── AGENTS.md              # 本ファイル（Agent 向けルール）
├── project_context.md     # プロジェクト要件（正）
├── future_extensions.md   # 将来拡張バックログ
├── README.md              # セットアップ・実行手順
├── .env.example           # 環境変数テンプレート
├── src/                   # 本番コード（src レイアウト）
│   ├── main.py            # オーケストレーター
│   ├── config.py          # 環境変数の一元管理
│   ├── exceptions.py      # ドメイン例外
│   ├── http_client.py     # HTTP 共通処理
│   ├── audio_utils.py     # 音声・ログ共通ユーティリティ
│   ├── listener.py
│   ├── retriever.py
│   ├── composer.py
│   ├── brain.py
│   ├── speaker.py
│   └── status_led.py
└── tests/                 # ミラーテスト（src と 1:1 対応）
    ├── conftest.py
    └── test_*.py
```

## 絶対に守る制約

### RAG・回答品質

- **生成前 RAG** を維持する。検索 → プロンプト構成 → 生成の順序を崩さない
- `composer.py` のプロンプトは検索結果を「絶対に事実」として扱う指示を含める
- 回答は **日本語のみ**。アルファベット（英語単語・文）を含めない（Piper 安定性のため）

### 依存ライブラリ

| 用途 | 使用する | 使用しない |
|------|----------|------------|
| TTS | `piper-tts-plus` | 本家 Piper（espeak-ng 依存） |
| STT | `faster-whisper` | — |
| HTTP | `requests` | — |
| GPIO LED | `gpiozero`（Pi のみ） | 非 Pi での強制有効化 |
| テスト | `pytest`, `pytest-mock` | 実 API への接続 |

### ハードウェア・環境

- `status_led.py` / `gpiozero` は Raspberry Pi 上でのみ動作。非 Pi 環境では自動無効化する現行挙動を維持する
- 音声入出力は `arecord` / `aplay`（ALSA）を使用
- 環境変数は **必ず `src/config.py` に集約** する。各モジュールで `os.getenv()` を直接書かない
- `.env` は Git にコミットしない（`.env.example` を更新する）

### テスト

- **外部 API（SearXNG / Ollama）へ実リクエストを送らない**
- `pytest-mock` または `unittest.mock` でモック化する
- 新規 `src/*.py` には対応する `tests/test_*.py` を追加する（ミラーテスト）
- テスト実行: `python3 -m pytest tests/ -v`
- 外部依存（`faster_whisper`, `piper`, `gpiozero` 等）は import 前にモックする既存パターンに従う

## コーディング規約

### スタイル

- Python 3、モジュール先頭に `#!/usr/bin/env python3` と docstring
- クラスベースのコンポーネント設計（`Listener`, `Retriever`, `Composer`, `Brain`, `Speaker`, `StatusLED`）
- ログは `logging.getLogger(__name__)` を使用。`print` は初期化メッセージ等の既存箇所に合わせる
- 初期化ログは `audio_utils.log_init` / `log_ready` を使う

### エラーハンドリング

- ドメイン例外は `src/exceptions.py` に定義する
  - `ListenerError`, `SearchError`, `GenerationError`, `SpeakerError`
- HTTP 通信は `src/http_client.py` の `http_get_json` / `http_post_json` を使い、呼び出し側の `error_class` で例外を分ける
- URL は `config.validate_url()` で検証する
- `main.py` の方針に合わせる:
  - 検索失敗は **非致命**（空コンテキストで続行）
  - 録音・認識・生成・発話の失敗は **致命**（適切な例外を送出）

### 変更時の原則

- **最小スコープ**: 依頼された問題だけを直す。無関係なリファクタリングをしない
- **既存パターンを踏襲**: 周辺コードの命名・構造・テストスタイルに合わせる
- **過剰な抽象化を避ける**: 1〜2 行のヘルパーや不要なエラーハンドリングを増やさない
- **コメント**: 自明な処理には書かない。非自明なビジネスロジック・ハードウェア制約のみ

## 外部サービス

| サービス | 用途 | デフォルト URL |
|----------|------|----------------|
| SearXNG | Web 検索 | `http://localhost:8080` |
| Ollama | LLM 生成 | `http://localhost:11434/api/generate` |

モデル: Whisper `small`（CPU / int8）、Ollama `qwen2.5:3b`、Piper 日本語 ONNX（`models/` 配下、Git 管理外）

## Agent への作業指示

### 実装・修正を依頼されたとき

1. `project_context.md` と本ファイルの制約を確認する
2. 変更対象モジュールと対応テストをセットで扱う
3. 実装後に `python3 -m pytest tests/ -v` を実行し、結果を報告する
4. README / `.env.example` に影響がある変更はドキュメントも更新する

### 将来拡張（`future_extensions.md`）を実装するとき

- ウェイクワード、会話履歴、ローカル RAG 等は **明示的な依頼がある場合のみ** 着手する
- 既存の 5 ステップパイプラインを壊さない
- 新モジュールは `src/` に置き、テストを `tests/` に追加する

### やってはいけないこと

- 本家 Piper（espeak-ng）への差し替え
- テストから実 SearXNG / Ollama への接続
- `models/` や `.env` のコミット
- `config.py` を経由しない環境変数の直書き
- 英語混在プロンプトへの変更（TTS 品質劣化）
- 生成後 RAG への設計変更（事前検索 → プロンプト注入の原則を維持）

## よく使うコマンド

```bash
# テスト
python3 -m pytest tests/ -v

# 実行（Pi 上・依存サービス起動後）
python3 src/main.py

# 環境変数セットアップ
cp .env.example .env
```

## 参照ドキュメント

| ファイル | 内容 |
|----------|------|
| `project_context.md` | 要件・依存・実装注意点 |
| `README.md` | セットアップ・Docker・ディレクトリ構成 |
| `future_extensions.md` | 未実装の拡張ロードマップ |
| `.env.example` | 環境変数一覧 |
