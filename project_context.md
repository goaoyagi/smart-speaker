# AIスマートスピーカー開発要件（ラズパイ5）

## 1. システムアーキテクチャ（ローカルRAG構成）
ハルシネーション（嘘）を徹底的に防ぐため、以下の順序で処理を行う。生成後の検証ではなく、**「生成前RAG（あらかじめカンニングペーパーを渡す）」を徹底**すること。最終回答の生成は、検索とコンテキスト構成の後に**1回だけ**行う。クエリ書き換えなど、最終回答の生成より前の補助的なLLM呼び出しは順序違反とみなさない。

1. **[耳] listener.py**: Whisper.cpp でユーザーの音声をテキスト化。
2. **[検索] retriever.py**: 質問をトリガーに、ローカルの「SearXNG」でWeb検索を実行。
3. **[並べ替え] retriever.py**: 検索結果を日本語Reranker（Optimum/ONNX のクロスエンコーダ）により質問との関連度で並べ替え、上位のみを次段に渡す。
4. **[構成] composer.py**: Reranking後の検索結果（事実ソース）と質問をプロンプトに編成。
5. **[脳] brain.py**: プロンプトを Ollama（Qwen2.5:3b）に投入し、事実に基づく回答を生成。
6. **[口] speaker.py**: `piper-tts-plus` で音声合成して発話。
7. **[視覚] status_led.py**: GPIO接続のLEDで、待機・聞き取り・検索・思考・発話・エラーの各状態を表示。
8. **[操作] push_to_talk.py**: GPIO接続のボタンを押している間だけ録音するプッシュ・トゥ・トーク。
9. **[記憶] conversation_history.py**: 直近N回の「問いと答え」を保持し、再復唱と文脈を踏まえた深掘り質問に対応。

## 2. 必要依存ライブラリ
**依存の正は `pyproject.toml`**。以下は用途の説明であり、追加・変更時は `pyproject.toml` と `README.md` のセットアップ手順もあわせて更新すること。

### 開発・テスト用（`pyproject.toml` の `dev` extra）
- **pytest** : テスト駆動開発用のメインフレームワーク
- **pytest-mock** : 外部APIやハードウェアをモック化するためのプラグイン

### 本番ロジック用
- **faster-whisper** : 音声認識（Whisperの高速実装）
- **numpy** : 録音データの配列処理
- **piper-tts-plus** : 日本語特化の音声合成（OpenJTalk内蔵版）
- **requests** : SearXNGサーバーおよびOllamaローカルAPIとの通信用
- **optimum[onnxruntime]** : Reranking用ONNXモデルの実行基盤
- **transformers** : Rerankingモデルのトークナイズおよび推論
- **fugashi / unidic-lite** : 日本語テキストの形態素解析

### Raspberry Pi でのみ必要（`pyproject.toml` の `pi` extra）
- **gpiozero** : ステータスLEDとプッシュ・トゥ・トークボタンのGPIO制御。非Pi環境では不在を検知して自動的に無効化される

## 3. 各コンポーネントの実装注意点
### speaker.py (音声合成)
- 本家Piper（espeak-ng依存）は日本語のアクセント解析が未対応のため絶対に使用しないこと。
- 必ず日本語特化のフォーク版である `piper-tts-plus` を使用すること。
- 使用する音声モデルは、Hugging Face等からJVSデータセット等の日本語ONNXモデルとJSON設定ファイルを取得して利用すること。

### retriever.py / brain.py (外部通信)
- テスト実行時は、実際のローカルサーバー（SearXNG / Ollama）にリクエストを飛ばさず、必ず `pytest-mock` を使用してレスポンスをシミュレート（モック化）すること。

### retriever.py (Reranking)
- Reranking を標準フローの一部（既定で有効）とし、生成前RAGの処理順序（検索 → Reranking → プロンプト構成 → 生成）を崩さないこと。
- `optimum` / `transformers` は動的インポート（使用直前に import し、`ImportError` を捕捉）すること。起動時の依存不足による失敗を避け、フォールバックできるようにする。
- 依存ライブラリやモデルのロード・推論に失敗した場合は Reranking をスキップして検索結果をそのまま返し、検索処理自体は継続すること（Reranking の失敗を非致命とする）。
- ユニットテストではモデルをダウンロード・ロードしない。Reranking を無効化するか、Reranker を `pytest-mock` でモック化し、
  オフライン（外部アクセス禁止）のテスト方針を守ること。

### status_led.py (ステータスLED)
- 待機・聞き取り・検索・思考・発話・エラーの状態を、`gpiozero` の `LED` の点灯・消灯・点滅パターンで表示する。点灯パターンは `LedState` で定義する。
- `gpiozero` の import 失敗や実行時失敗は握りつぶして処理を継続すること。非Pi環境やGPIO未接続時にLEDが原因で会話が止まってはならない。
- `STATUS_LED_ENABLED` で無効化できるようにする。ピン番号は `STATUS_LED_PIN` で設定する。

### push_to_talk.py (プッシュ・トゥ・トーク)
- `gpiozero` の `Button` の押下（`when_pressed`）／解放（`when_released`）を検出し、押している間だけ録音する。`listener.py` は `start_recording` / `stop_recording` で開始・停止を制御する。
- ボタンが利用できない環境（`gpiozero` 不在、`PUSH_TO_TALK_ENABLED=false`）では自動的に無効化し、`RECORD_SECONDS` の固定秒数録音にフォールバックすること（`status_led.py` と同じ方針）。
- チャタリング対策として `PTT_BOUNCE_TIME` を設定し、極端に短い押下や押しっぱなしは `PTT_MIN_RECORD_SECONDS` / `PTT_MAX_RECORD_SECONDS` でガードする。
- `main.py` のメインループは、ボタンが利用できる場合はイベント駆動（押下待ち→録音→解放で確定→処理）で動作する。

### conversation_history.py (会話コンテキストの保持)
- Sliding Window Memory として `collections.deque`（`maxlen=CONVERSATION_MAX_TURNS`）を使い、古い履歴をO(1)で自動破棄する。
- Condense Question として、保持している履歴を短い要約文字列に整形して `composer.py` のプロンプトへ埋め込む。プロンプト長を抑えるため、各回答は `CONVERSATION_ANSWER_CLIP` 文字で打ち切る。
- 「もう一回言って」などの再復唱コマンドを検知した場合は、検索とLLM生成を行わず直前の回答をそのまま発話する。
- 履歴が空のときは空文字列を返し、呼び出し側が条件分岐なしに埋め込めるようにする。

## 4. ディレクトリ構成（srcレイアウト・ミラーテスト）
```text
smart-speaker/
├── project_context.md      # 本ドキュメント
├── pyproject.toml          # 依存ライブラリの正
├── src/
│   ├── __init__.py
│   ├── main.py             # 全体を統括するオーケストレーター
│   ├── listener.py
│   ├── retriever.py
│   ├── composer.py
│   ├── brain.py
│   ├── speaker.py
│   ├── status_led.py
│   ├── push_to_talk.py
│   ├── conversation_history.py
│   ├── config.py           # 環境変数の一元管理・URL検証
│   ├── http_client.py      # 共通HTTPクライアント
│   ├── exceptions.py       # ドメイン固有の例外
│   └── audio_utils.py      # 音声・ログ共通ユーティリティ
└── tests/
    ├── __init__.py
    ├── conftest.py         # pytest共通フィクスチャ・モック定義
    ├── test_main.py
    ├── test_listener.py
    ├── test_retriever.py
    ├── test_composer.py
    ├── test_brain.py
    ├── test_speaker.py
    ├── test_status_led.py
    ├── test_push_to_talk.py
    ├── test_conversation_history.py
    ├── test_config.py
    ├── test_http_client.py
    └── test_audio_utils.py
```
