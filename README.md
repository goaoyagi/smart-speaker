# AI Smart Speaker (Raspberry Pi 5)

ローカルRAG構成のAIスマートスピーカー。ハルシネーションを防ぐため、生成前RAG（検索結果を事前にプロンプトに含める）を採用。

## アーキテクチャ

1. **[耳] listener.py**: Whisper.cpp でユーザーの音声をテキスト化
2. **[検索] retriever.py**: 質問をトリガーに、ローカルの「SearXNG」でWeb検索を実行
3. **[構成] composer.py**: 検索結果（事実ソース）と質問をプロンプトに編成
4. **[脳] brain.py**: プロンプトを Ollama（Qwen2.5:3b）に投入し、事実に基づく回答を生成
5. **[口] speaker.py**: `piper-tts-plus` で音声合成して発話
6. **[視覚] status_led.py**: GPIO接続のLEDで、待機・聞き取り・検索・思考・発話・エラーを表示
7. **[操作] push_to_talk.py**: GPIO接続のボタンを押している間だけ録音するプッシュ・トゥ・トーク

## 必要依存ライブラリ

### 開発・テスト用
```bash
sudo apt install python3-pytest python3-pytest-mock python3-numpy python3-requests
```

### 本番ロジック用
```bash
pip install faster-whisper piper-tts-plus requests
```

または既存の仮想環境を使用:
```bash
# 仮想環境をアクティベート
source /path/to/your/venv/bin/activate
python3 src/main.py
```

## セットアップ

1. **環境変数の設定**
   ```bash
   cp .env.example .env
   # .envファイルを編集して環境に合わせて設定
   ```

2. **Dockerコンテナの起動**
   ```bash
   # SearXNG
   docker run -d -p 8080:8080 --name searxng searxng/searxng
   
   # Ollama (ローカル実行の場合は不要)
   docker run -d -p 11434:11434 --name ollama ollama/ollama
   ```

3. **音声モデルの配置**
   ```bash
   # Piper TTSモデルをプロジェクトルートのmodels/ディレクトリに配置
   mkdir -p models
   # モデルファイルを models/ に配置
   ```

## 実行

```bash
# 仮想環境を使用
source venv/bin/activate
python3 src/main.py

# またはシステムPythonを使用
python3 src/main.py
```

## テスト

```bash
python3 -m pytest tests/ -v
```

## ディレクトリ構成

```
smart-speaker/
├── AGENTS.md               # AIエージェント向け開発ルール
├── project_context.md      # プロジェクト要件
├── future_extensions.md    # 将来の機能拡張バックログ
├── README.md               # 本ファイル（セットアップ・実行方法）
├── .gitignore              # Git除外設定
├── .env.example            # 環境変数テンプレート
├── conftest.py             # src/ を sys.path に追加（テスト用）
├── src/
│   ├── __init__.py
│   ├── main.py             # オーケストレーター
│   ├── listener.py         # Whisper音声認識
│   ├── retriever.py        # SearXNG Web検索
│   ├── composer.py         # RAGプロンプト構成
│   ├── brain.py            # Ollama AI生成
│   ├── speaker.py          # Piper-Plus TTS
│   ├── status_led.py       # GPIO ステータスLED制御
│   ├── push_to_talk.py     # GPIOボタンによるプッシュ・トゥ・トーク
│   ├── config.py           # 環境変数の一元管理・URL検証
│   ├── http_client.py      # 共通HTTPクライアント
│   ├── exceptions.py       # ドメイン固有の例外
│   └── audio_utils.py      # 音声・ログ共通ユーティリティ
└── tests/
    ├── __init__.py
    ├── conftest.py         # pytest共通フィクスチャ
    ├── test_main.py
    ├── test_listener.py
    ├── test_retriever.py
    ├── test_composer.py
    ├── test_brain.py
    ├── test_speaker.py
    ├── test_status_led.py
    ├── test_push_to_talk.py
    ├── test_config.py
    ├── test_http_client.py
    └── test_audio_utils.py
```

## 注意点

- **speaker.py**: 本家Piper（espeak-ng依存）は日本語のアクセント解析が未対応のため、必ず `piper-tts-plus` を使用すること
- **status_led.py**: `gpiozero` は Raspberry Pi 上でのみ動作するため、非Pi環境では自動的に無効化される
- **push_to_talk.py**: GPIOボタンが利用できる環境では「押している間だけ録音」するプッシュ・トゥ・トークで動作する。ボタンが無い非Pi環境では自動的に無効化され、`RECORD_SECONDS` の固定秒数録音にフォールバックする。`PTT_MIN_RECORD_SECONDS` / `PTT_MAX_RECORD_SECONDS` で最小・最大録音時間を制御する
- **テスト実行時**: 外部API（SearXNG/Ollama）にリクエストを飛ばさず、`pytest-mock` でモック化すること
