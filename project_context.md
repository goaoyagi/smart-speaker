# AIスマートスピーカー開発要件（ラズパイ5）

## 1. システムアーキテクチャ（ローカルRAG構成）
ハルシネーション（嘘）を徹底的に防ぐため、以下の順序で処理を行う。生成後の検証ではなく、**「生成前RAG（あらかじめカンニングペーパーを渡す）」を徹底**すること。最終回答の生成は、検索とコンテキスト構成の後に**1回だけ**行う。クエリ書き換えなど、最終回答の生成より前の補助的なLLM呼び出しは順序違反とみなさない。

1. **[耳] listener.py**: Whisper.cpp でユーザーの音声をテキスト化。
2. **[検索準備] query_prep.py**: 検索要否判定と、指示語を含む続き質問のクエリ書き換え（生成前の補助LLM呼び出し。失敗時は元の質問で検索する）。
3. **[検索] retriever.py**: 質問をトリガーに、ローカルの「SearXNG」でWeb検索を実行（不要ならスキップ）。
4. **[並べ替え] retriever.py**: 検索結果を日本語Reranker（Optimum/ONNX のクロスエンコーダ）により質問との関連度で並べ替え、上位のみを次段に渡す。
5. **[本文取得・二段Rerank] retriever.py**: 並べ替え後の上位URLを並列取得し、`trafilatura` で本文抽出（失敗はスニペットへ）。本文は段落・文境界（句点/改行/`！？!?`）を優先しつつ約`PASSAGE_CHARS`字のパッセージに分割し（句読点のない長文は`PASSAGE_CHARS`字で強制的に分割し、1パッセージが上限を超えないようにする）、既存Rerankerで質問との関連度により再度並べ替える。`RERANK_MIN_SCORE`未満のパッセージ・結果は落とす（既定0.0では足切りしない）。最終的に残ったパッセージの合計文字数が`CONTEXT_CHAR_BUDGET`を超える場合は関連度の低いものから後ろへ落とす（1件も無くなることはない）。`FETCH_PAGE_ENABLED=false`で現行のスニペット経路に戻せる。
6. **[構成] composer.py**: 本文取得・Reranking後の検索結果（事実ソース）と質問をプロンプトに編成。
7. **[脳] brain.py**: プロンプトを Ollama（Qwen2.5:3b）に投入し、事実に基づく回答を生成。
8. **[口] speaker.py**: `piper-tts-plus` で音声合成して発話。単位や略語の英字は発話前に日本語読みへ正規化する。
9. **[視覚] status_led.py**: GPIO接続のLEDで、待機・聞き取り・検索・思考・発話・エラーの各状態を表示。
10. **[操作] push_to_talk.py**: GPIO接続のボタンを押している間だけ録音するプッシュ・トゥ・トーク。
11. **[記憶] conversation_history.py**: 直近N回の「問いと答え」を保持し、再復唱と文脈を踏まえた深掘り質問に対応。

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
- **trafilatura** : 検索結果ページのHTML本文抽出（retriever.py）

### Raspberry Pi でのみ必要（`pyproject.toml` の `pi` extra）
- **gpiozero** : ステータスLEDとプッシュ・トゥ・トークボタンのGPIO制御。非Pi環境では不在を検知して自動的に無効化される

## 3. 各コンポーネントの実装注意点
### speaker.py (音声合成)
- 本家Piper（espeak-ng依存）は日本語のアクセント解析が未対応のため絶対に使用しないこと。
- 必ず日本語特化のフォーク版である `piper-tts-plus` を使用すること。
- 使用する音声モデルは、Hugging Face等からJVSデータセット等の日本語ONNXモデルとJSON設定ファイルを取得して利用すること。
- 発話前に `speech_normalize.normalize_for_speech()` で単位・略語の英字を日本語読みへ置換する（呼び出しは `text_turn.py` で行う）。

### retriever.py / brain.py (外部通信)
- テスト実行時は、実際のローカルサーバー（SearXNG / Ollama）にリクエストを飛ばさず、必ず `pytest-mock` を使用してレスポンスをシミュレート（モック化）すること。

### brain.py (生成パラメータ)
- Ollama の生成パラメータ（`OLLAMA_NUM_CTX` / `OLLAMA_NUM_PREDICT` / `OLLAMA_TEMPERATURE` / `OLLAMA_REPEAT_PENALTY` / `OLLAMA_KEEP_ALIVE`）と、日本語限定を担保する `OLLAMA_SYSTEM_PROMPT` は `.env` で指定できるようにする。検索要否判定・クエリ書き換えは `QUERY_PREP_ENABLED` / `OLLAMA_AUX_NUM_PREDICT` / `OLLAMA_AUX_TEMPERATURE` で制御する。既定値は `src/config.py` に置く。
- `/api/chat` に `messages` 配列と `keep_alive` / `options` を渡す。補助呼び出しは `generate_auxiliary()` を使い、spoken system prompt は付けない。
- `prompt[:10000]` の文字切り捨ては使わない。検索コンテキストは `CONTEXT_CHAR_BUDGET`（既定2000字）に収まるよう検索結果単位で後ろから落とし、messages 全体の概算トークンが `OLLAMA_NUM_CTX - OLLAMA_NUM_PREDICT` を超える場合は古い履歴ターンから落とす。system と最後の user メッセージは落とさない。
- 文字列を渡す旧 `generate_response(prompt)` 経路は残し、`main.py` から `compose_prompt` へ切り戻せるようにする。

### retriever.py (Reranking)
- Reranking を標準フローの一部（既定で有効）とし、生成前RAGの処理順序（検索 → Reranking → プロンプト構成 → 生成）を崩さないこと。
- `optimum` / `transformers` は動的インポート（使用直前に import し、`ImportError` を捕捉）すること。起動時の依存不足による失敗を避け、フォールバックできるようにする。
- 依存ライブラリやモデルのロード・推論に失敗した場合は Reranking をスキップして検索結果をそのまま返し、検索処理自体は継続すること（Reranking の失敗を非致命とする）。
- ユニットテストではモデルをダウンロード・ロードしない。Reranking を無効化するか、Reranker を `pytest-mock` でモック化し、
  オフライン（外部アクセス禁止）のテスト方針を守ること。

### retriever.py (本文取得・二段Rerank)
- `FETCH_PAGE_ENABLED=true`（既定）のとき、結果単位Rerank後の上位`FETCH_PAGE_TOP_N`件のURLを並列取得し、`trafilatura`で本文抽出する。`trafilatura`も動的インポートにすること。
- 本文取得・抽出の失敗は非致命。その件だけ元のスニペットへ戻し、`logger.warning`を残して処理を続ける（検索全体は`SearchError`にしない）。HTTP取得は`src.http_client.http_get_text`に集約し、`requests`を直接呼ばない。
- 抽出した本文は約`PASSAGE_CHARS`字のパッセージに分割し、既存Rerankerで質問との関連度により再度並べ替える（モデルの再ロードはしない）。分割は段落（改行）→文（句点・`！？!?`・改行）の順で境界を優先し、句読点のない1文が`PASSAGE_CHARS`字を超える場合はその文自体を`PASSAGE_CHARS`字ずつ強制的にチャンク分割する。1パッセージが上限を超えることは無い。
- `RERANK_MIN_SCORE`未満のスコアの結果・パッセージは落とす。全件落ちた場合は空リストを返し、composerには「検索結果なし」として渡す。
- 二段Rerank後の最終パッセージ群は、`title`+`content`の合計文字数が`CONTEXT_CHAR_BUDGET`に収まるよう関連度の低い方（リストの後ろ）から落とす。ただし1件のみで既に上限を超える場合でもその1件は残す（結果を空にしない）。`composer.clip_search_results`（プロンプト構成時のクリップ）とは独立した安全策であり、`FETCH_PAGE_TOP_N`件×`PASSAGE_CHARS`字の細かいパッセージが多数になっても、retriever が返す時点で総量を有限に保つ。
- `FETCH_PAGE_ENABLED=false`で本文取得をスキップし、現行のスニペット経路に戻せること。

### composer.py (プロンプト構成)
- 本番経路は `compose_messages()`。フェーズ2で確定した指示は system メッセージに集約する。
- 検索結果は `CONTEXT_CHAR_BUDGET` に収まるよう、検索結果の単位で後ろから落とす。
- 検索結果は「質問に関係するもの」だけを事実として扱い、無関係なら使わず、使う場合は推測で補わない。
- 指示語（「それ」「さっきの」）は会話履歴を参照して解釈する。質問は最後の user メッセージ末尾に置く。
- 文数は質問タイプに合わせる（事実は1〜3文、挨拶・不明は1〜2文、説明は3〜5文）。結論先出し。発話に「検索結果」とは言わない。
- `compose_prompt()` は切り戻し用に残す。検索結果あり／なしの両分岐で、日本語限定と上記の文数方針を指示する。

### status_led.py (ステータスLED)
- 待機・聞き取り・検索・思考・発話・エラーの状態を、`gpiozero` の `LED` の点灯・消灯・点滅パターンで表示する。点灯パターンは `LedState` で定義する。
- `gpiozero` の import 失敗や実行時失敗は握りつぶして処理を継続すること。非Pi環境やGPIO未接続時にLEDが原因で会話が止まってはならない。
- `STATUS_LED_ENABLED` で無効化できるようにする。ピン番号は `STATUS_LED_PIN` で設定する。

### push_to_talk.py (プッシュ・トゥ・トーク)
- `gpiozero` の `Button` の押下（`when_pressed`）／解放（`when_released`）を検出し、押している間だけ録音する。`listener.py` は `start_recording` / `stop_recording` で開始・停止を制御する。
- ボタンが利用できない環境（`gpiozero` 不在、`PUSH_TO_TALK_ENABLED=false`）では自動的に無効化し、`RECORD_SECONDS` の固定秒数録音にフォールバックすること（`status_led.py` と同じ方針）。
- チャタリング対策として `PTT_BOUNCE_TIME` を設定し、極端に短い押下や押しっぱなしは `PTT_MIN_RECORD_SECONDS` / `PTT_MAX_RECORD_SECONDS` でガードする。
- `main.py` のメインループは、ボタンが利用できる場合はイベント駆動（押下待ち→録音→解放で確定→処理）で動作する。

### query_prep.py (検索準備)
- 最終回答の前に、検索要否判定とクエリ書き換えを1回の補助LLM呼び出しで行う。
- 失敗は非致命。元の質問で検索する。結果は発話しない。
- `QUERY_PREP_ENABLED=false` のときは補助呼び出しをせず、常に元の質問で検索する。

### speech_normalize.py / speaker.py (発話正規化)
- 生成後に LLM で推敲せず、単位・略語の英字を日本語読みへ機械的に置換してから Piper に渡す。
- `main.py` と評価ハーネスは同じ `text_turn.py` を使い、正規化後の文を回答として扱う。

### conversation_history.py (会話コンテキストの保持)
- Sliding Window Memory として `collections.deque`（`maxlen=CONVERSATION_MAX_TURNS`）を使い、古い履歴をO(1)で自動破棄する。
- Condense Question として、保持している履歴を短い要約文字列に整形して `composer.py` のプロンプトへ埋め込む。プロンプト長を抑えるため、各回答は `CONVERSATION_ANSWER_CLIP` 文字で打ち切る。
- `/api/chat` 経路では `as_messages()` で user / assistant の交互メッセージを返す。`as_condensed_context()` は `compose_prompt()` 経路の後方互換のため残す。
- 「もう一回言って」などの再復唱コマンドを検知した場合は、検索とLLM生成を行わず直前の回答をそのまま発話する。
- 履歴が空のときは空文字列を返し、呼び出し側が条件分岐なしに埋め込めるようにする。

### 評価ハーネス
- `scripts/eval.py` と `scripts/eval_cases.json` は、改善の前後を数値で比較するための手動評価用ハーネスである。SearXNG と Ollama に接続して実サービスを評価するため、pytest の対象にはしない。本番と同じ `run_text_turn`（検索準備 → 検索 → 構成 → 生成 → 発話正規化）を通す。

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
│   ├── query_prep.py
│   ├── text_turn.py
│   ├── speech_normalize.py
│   ├── brain.py
│   ├── speaker.py
│   ├── status_led.py
│   ├── push_to_talk.py
│   ├── conversation_history.py
│   ├── config.py           # 環境変数の一元管理・URL検証
│   ├── http_client.py      # 共通HTTPクライアント
│   ├── exceptions.py       # ドメイン固有の例外
│   └── audio_utils.py      # 音声・ログ共通ユーティリティ
├── scripts/                # 手動実行スクリプト（pytest対象外）
│   ├── button_led_test.py
│   ├── eval.py             # 回答精度の手動評価ハーネス
│   └── eval_cases.json     # 評価ケース定義
└── tests/
    ├── __init__.py
    ├── conftest.py         # pytest共通フィクスチャ・モック定義
    ├── test_main.py
    ├── test_listener.py
    ├── test_retriever.py
    ├── test_composer.py
    ├── test_query_prep.py
    ├── test_text_turn.py
    ├── test_speech_normalize.py
    ├── test_brain.py
    ├── test_speaker.py
    ├── test_status_led.py
    ├── test_push_to_talk.py
    ├── test_conversation_history.py
    ├── test_config.py
    ├── test_http_client.py
    ├── test_audio_utils.py
    ├── test_logging_policy.py
    └── test_eval_harness.py
```
