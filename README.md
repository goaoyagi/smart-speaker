# AI Smart Speaker (Raspberry Pi 5)

ローカルRAG構成のAIスマートスピーカー。ハルシネーションを防ぐため、生成前RAG（検索結果を事前にプロンプトに含める）を採用。

## アーキテクチャ

1. **[耳] listener.py**: Whisper.cpp でユーザーの音声をテキスト化
2. **[検索] retriever.py**: 質問をトリガーに、ローカルの「SearXNG」でWeb検索を実行
3. **[構成] composer.py**: 検索結果（事実ソース）と質問をプロンプトに編成
4. **[脳] brain.py**: プロンプトを Ollama（Qwen2.5:3b）に投入し、事実に基づく回答を生成
5. **[口] speaker.py**: `piper-tts-plus` で音声合成して発話

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
├── project_context.md      # プロジェクト要件
├── README.md              # 本ファイル
├── .gitignore            # Git除外設定
├── .env.example          # 環境変数テンプレート
├── src/
│   ├── __init__.py
│   ├── main.py           # オーケストレーター
│   ├── listener.py       # Whisper音声認識
│   ├── retriever.py      # SearXNG Web検索
│   ├── composer.py       # RAGプロンプト構成
│   ├── brain.py          # Ollama AI生成
│   └── speaker.py        # Piper-Plus TTS
└── tests/
    ├── __init__.py
    ├── conftest.py       # pytest共通フィクスチャ
    ├── test_main.py
    ├── test_listener.py
    ├── test_retriever.py
    ├── test_composer.py
    ├── test_brain.py
    └── test_speaker.py
```

## 注意点

- **speaker.py**: 本家Piper（espeak-ng依存）は日本語のアクセント解析が未対応のため、必ず `piper-tts-plus` を使用すること
- **テスト実行時**: 外部API（SearXNG/Ollama）にリクエストを飛ばさず、`pytest-mock` でモック化すること
