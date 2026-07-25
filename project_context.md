# AIスマートスピーカー開発要件（ラズパイ5）

## 1. システムアーキテクチャ（ローカルRAG構成）
ハルシネーション（嘘）を徹底的に防ぐため、以下の順序で処理を行う。生成後の検証ではなく、**「生成前RAG（あらかじめカンニングペーパーを渡す）」を徹底**すること。

1. **[耳] listener.py**: Whisper.cpp でユーザーの音声をテキスト化。
2. **[検索] retriever.py**: 質問をトリガーに、ローカルの「SearXNG」でWeb検索を実行。
   検索結果は、任意（オプション）で日本語Reranker（Optimum/ONNX のクロスエンコーダ）により質問との関連度で並べ替え、
   上位のみを次段に渡す。Reranking は無効化可能で、無効時・失敗時は SearXNG の順序をそのまま使う。
3. **[構成] composer.py**: 検索結果（事実ソース）と質問をプロンプトに編成。
4. **[脳] brain.py**: プロンプトを Ollama（Qwen2.5:3b）に投入し、事実に基づく回答を生成。
5. **[口] speaker.py**: `piper-tts-plus` で音声合成して発話。

## 2. 必要依存ライブラリ
開発環境および本番環境の構築時、以下のライブラリを `pip install` すること。

### 開発・テスト用
- **pytest** : テスト駆動開発用のメインフレームワーク
- **pytest-mock** : 外部APIやハードウェアをモック化するためのプラグイン

### 本番ロジック用
- **piper-tts-plus** : 日本語特化の音声合成（OpenJTalk内蔵版）
- **requests** : SearXNGサーバーおよびOllamaローカルAPIとの通信用

## 3. 各コンポーネントの実装注意点
### speaker.py (音声合成)
- 本家Piper（espeak-ng依存）は日本語のアクセント解析が未対応のため絶対に使用しないこと。
- 必ず日本語特化のフォーク版である `piper-tts-plus` を使用すること。
- 使用する音声モデルは、Hugging Face等からJVSデータセット等の日本語ONNXモデルとJSON設定ファイルを取得して利用すること。

### retriever.py / brain.py (外部通信)
- テスト実行時は、実際のローカルサーバー（SearXNG / Ollama）にリクエストを飛ばさず、必ず `pytest-mock` を使用してレスポンスをシミュレート（モック化）すること。

### retriever.py (Reranking)
- Reranking はあくまで任意機能。既定は無効とし、生成前RAGの処理順序（検索 → 並べ替え → プロンプト構成 → 生成）を崩さないこと。
- `optimum` / `transformers` は動的インポート（使用直前に import し、`ImportError` を捕捉）すること。
  未インストール環境でも起動・検索が失敗しないようにし、その場合は Reranking をスキップして検索結果をそのまま返す。
- モデルのロード・推論に失敗した場合も同様にスキップし、検索処理自体は継続すること（Reranking の失敗を致命扱いにしない）。
- ユニットテストではモデルをダウンロード・ロードしない。Reranking を無効化するか、Reranker を `pytest-mock` でモック化し、
  オフライン（外部アクセス禁止）のテスト方針を守ること。

## 4. ディレクトリ構成（srcレイアウト・ミラーテスト）
```text
smart-speaker/
├── project_context.md      # 本ドキュメント
├── src/
│   ├── __init__.py
│   ├── main.py             # 全体を統括するオーケストレーター
│   ├── listener.py
│   ├── retriever.py
│   ├── composer.py
│   ├── brain.py
│   └── speaker.py
└── tests/
    ├── __init__.py
    ├── conftest.py         # pytest共通フィクスチャ・モック定義
    ├── test_main.py
    ├── test_listener.py
    ├── test_retriever.py
    ├── test_composer.py
    ├── test_brain.py
    └── test_speaker.py
