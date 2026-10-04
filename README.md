# AI Smart Speaker (Raspberry Pi 5)

ローカルRAG構成のAIスマートスピーカー。ハルシネーションを防ぐため、生成前RAG（検索結果を事前にプロンプトに含める）を採用。

## アーキテクチャ

1. **[耳] listener.py**: Whisper.cpp でユーザーの音声をテキスト化
2. **[検索準備] query_prep.py**: 検索要否判定と、指示語を含む続き質問のクエリ書き換え（生成前の補助LLM呼び出し。失敗時は元の質問で検索）
3. **[検索] retriever.py**: 質問をトリガーに、ローカルの「SearXNG」でWeb検索を実行（不要ならスキップ）
4. **[並べ替え] retriever.py**: 日本語Reranker（Optimum/ONNX のクロスエンコーダ）で検索結果を質問との関連度順に並べ替え
5. **[本文取得・二段Rerank] retriever.py**: 上位URLを並列取得し `trafilatura` で本文抽出（失敗はスニペットへ）。本文を段落・文境界を優先しつつパッセージに分割し（句読点の無い長文は強制分割）、Rerankerで再度並べ替え。最終パッセージの合計文字数は `CONTEXT_CHAR_BUDGET` に収まるよう関連度の低い方から間引く
6. **[構成] composer.py**: 本文取得・Reranking後の検索結果（事実ソース）と質問をプロンプトに編成
7. **[脳] brain.py**: プロンプトを Ollama（Qwen2.5:3b）に投入し、事実に基づく回答を生成
8. **[口] speaker.py**: `piper-tts-plus` で音声合成して発話
9. **[視覚] status_led.py**: GPIO接続のLEDで、待機・聞き取り・検索・思考・発話・エラーを表示
10. **[操作] push_to_talk.py**: GPIO接続のボタンを押している間だけ録音するプッシュ・トゥ・トーク
11. **[記憶] conversation_history.py**: 直近N回の問いと答えを保持し、再復唱と文脈を踏まえた深掘り質問に対応

## 必要依存ライブラリ

依存ライブラリは `pyproject.toml` で管理しています。

### 本番ロジック用
```bash
pip install .
```

### 開発・テスト用
```bash
pip install -e ".[dev]"
```

### Raspberry Pi（GPIO を使う場合）
```bash
pip install ".[pi]"
```

既存の仮想環境を使う場合:
```bash
source venv/bin/activate
pip install -e ".[dev]"
```

## セットアップ

1. **環境変数の設定**
   ```bash
   cp .env.example .env
   # .envファイルを編集して環境に合わせて設定
   ```

   既存の `.env` を使っている場合、`OLLAMA_API_URL` を `http://localhost:11434/api/generate` から `http://localhost:11434/api/chat` に書き換えてください。既定値は `/api/chat` です。

2. **Dockerコンテナの起動**
   ```bash
   # SearXNG（JSON APIを使うため deploy/searxng/settings.yml が必要）
   docker run -d -p 8080:8080 --name searxng \
     -v "$(pwd)/deploy/searxng:/etc/searxng:ro" searxng/searxng
   
   # Ollama (ローカル実行の場合は不要)
   docker run -d -p 11434:11434 --name ollama ollama/ollama
   ```

   SearXNG の既定設定では `format=json` が拒否されるため、`retriever.py` から使うには `search.formats` に `json` を含む設定ファイルが必要です。Raspberry Pi で常駐運用する場合は、この手動起動ではなく「[Raspberry Pi での自動起動](#raspberry-pi-での自動起動常駐運用)」の `deploy/install.sh` を使ってください（`secret_key` の生成と systemd 登録まで行います）。

3. **音声モデルの配置**

   Piper TTS の音声モデル（日本語ONNX）とその設定ファイルを `models/piper/` に配置します。
   ```bash
   mkdir -p models/piper
   # models/piper/tsukuyomi.onnx      … 音声モデル本体
   # models/piper/tsukuyomi.onnx.json … 対応する設定ファイル
   ```

   設定ファイルは Piper の慣例に従い、モデル名に `.json` を付けた名前にします。
   別のモデルを使う場合は `.env` の `PIPER_MODEL_PATH` / `PIPER_CONFIG_PATH` で指定してください。

4. **Rerankingモデルの配置**

   Rerankingモデルを `models/reranker/` に配置します。
   インターネット非接続環境では、インターネット接続のある別マシンで `optimum-cli` をインストールし、ONNXモデルをエクスポートします。
   ```bash
   pip install optimum[onnxruntime] transformers fugashi unidic-lite
   optimum-cli export onnx \
     --model hotchpotch/japanese-reranker-cross-encoder-xsmall-v1 \
     --task text-classification \
     models/reranker
   ```

   エクスポートした `models/reranker/` ディレクトリを、そのままRaspberry Piのプロジェクトルートへコピーしてください。
   `models/` は `.gitignore` で除外されているため、モデルファイルはGit管理対象になりません。

## 実行

相対インポートを使用しているため、モジュール形式（`-m`）で実行する必要があります。

```bash
# 仮想環境をアクティベート
source venv/bin/activate
python3 -m src.main

# またはシステムPythonを使用
python3 -m src.main
```

## Raspberry Pi での自動起動（常駐運用）

ラズパイの電源を入れるだけで、SearXNGコンテナと `src/main.py` が起動し、ボタンを押せば会話できる状態にします。設定は `deploy/` にまとめてこのリポジトリで管理しています。

```
deploy/
├── install.sh              # systemdユニットの生成・配置・有効化
├── uninstall.sh            # ユニットの停止・削除
├── searxng.service         # SearXNGコンテナ用ユニットのテンプレート
├── smart-speaker.service   # python3 -m src.main 常駐用ユニットのテンプレート
└── searxng/settings.yml    # SearXNGのJSON API有効化設定（secret_keyは生成時に埋め込み）
```

### インストール

前提として、Raspberry Pi に Docker（`curl -fsSL https://get.docker.com | sh`）と Ollama がインストール済みで、`.env` とモデル（`models/piper/`・`models/reranker/`）を配置済みであることを確認してください。

```bash
cd ~/smart-speaker            # リポジトリのルート
sudo ./deploy/install.sh
```

`install.sh` は次の処理を行います。

1. `deploy/searxng/settings.yml` を `/etc/searxng/settings.yml` に配置（`secret_key` はランダム生成。既存ファイルがあれば上書きしない）
   - すでに動作している SearXNG の設定ファイルを流用する場合は、そのディレクトリを `SEARXNG_CONFIG_DIR` で指定する（例: `sudo SEARXNG_CONFIG_DIR=/home/pi/searxng ./deploy/install.sh`）。指定したディレクトリの `settings.yml` はそのまま使われる
   - service化そのものがJSON設定を要求するわけではない（`systemctl` は `docker run` の起動方法を変えるだけ）。`searxng/searxng` の現行イメージは設定なしだと `format=json` に 403 を返すため、設定ファイルの用意はコンテナの起動方法とは無関係に必要
2. テンプレートからパス・実行ユーザー・ポートを埋め込んだ `searxng.service` / `smart-speaker.service` を `/etc/systemd/system/` に配置
3. 両ユニットを `enable`（＝ラズパイ起動時に自動実行）し、起動する

主な上書き用の環境変数（既定値で問題なければ指定不要）:

```bash
sudo RUN_USER=pi PYTHON_BIN=/home/pi/smart-speaker/venv/bin/python3 SEARXNG_PORT=8080 ./deploy/install.sh
sudo START_SERVICES=false ./deploy/install.sh   # 配置と有効化のみ行い、起動はしない
```

実行ユーザーは既定で `sudo` を実行したユーザー（`$SUDO_USER`）、Python は `venv/bin/python3` があればそれ、無ければ `python3` を使います。`.env` はユニットの `WorkingDirectory`（リポジトリのルート）から `src/config.py` が読み込みます。

`.env` に `SEARXNG_URL` が無い場合、`install.sh` はユニットに `SEARXNG_URL=http://localhost:<SEARXNG_PORT>` を設定し、本体の検索先を公開ポートに合わせます。`.env` に `SEARXNG_URL` がある場合はその値を優先し、ポートが `SEARXNG_PORT` と一致しなければ警告を表示します。

上書き用の値は systemd ユニットにそのまま埋め込まれるため、`install.sh` は書式を検証し、空白・改行・`|` などを含む値は拒否します。

`smart-speaker.service` は `Restart=on-failure` です。異常終了した場合だけ再起動します。GPIO ボタンが使えない環境では、本体は1回だけ固定秒数の録音をして正常終了し、再起動しません。

Ollama をローカルにインストールしている場合、`ollama` のサービスも自動起動になっているか確認してください。

```bash
sudo systemctl enable --now ollama
```

アンインストール（ユニットの停止・削除。SearXNG設定も消す場合は `REMOVE_CONFIG=true`。ただし削除されるのは `install.sh` が生成した `settings.yml` だけで、流用した既存の設定やディレクトリ内のほかのファイルは残ります）:

```bash
sudo ./deploy/uninstall.sh
```

### 常駐プロセスの起動・停止

本体（`src/main.py`）:

```bash
sudo systemctl stop smart-speaker.service      # 停止
sudo systemctl start smart-speaker.service     # 常駐で起動
sudo systemctl restart smart-speaker.service   # 再起動（コード・.env の変更を反映）
sudo systemctl status smart-speaker.service    # 稼働状態の確認
sudo systemctl disable smart-speaker.service   # 起動時の自動実行をやめる
sudo systemctl enable smart-speaker.service    # 起動時の自動実行に戻す
```

SearXNG コンテナ:

```bash
sudo systemctl stop searxng.service       # コンテナを停止（--rm なので削除される）
sudo systemctl start searxng.service      # コンテナを起動
sudo systemctl restart searxng.service    # 作り直して起動
sudo systemctl status searxng.service     # 稼働状態の確認
docker ps                                 # コンテナの確認
curl "http://localhost:8080/search?q=test&format=json"   # JSON APIの疎通確認
```

`docker run` / `docker stop` を直接使わず `systemctl` で操作してください。ユニットは `Restart=always` なので、`docker stop` だけではすぐに再起動されます。

デバッグのために手動実行したいときは、常駐を止めてから実行します:

```bash
sudo systemctl stop smart-speaker.service
python3 -m src.main
```

### ログの確認

ログは systemd の journal に出ます（`logging` の出力がそのまま入ります）。

```bash
journalctl -u smart-speaker.service -f              # 本体のログを追尾
journalctl -u smart-speaker.service -n 200          # 直近200行
journalctl -u smart-speaker.service -b              # 今回の起動以降すべて
journalctl -u smart-speaker.service --since "10 min ago"
journalctl -u smart-speaker.service -p warning      # 警告以上だけ
journalctl -u searxng.service -f                    # SearXNGコンテナのログ
journalctl -u smart-speaker.service -u searxng.service -f   # 両方をまとめて追尾
```

## テスト

### ソフトウェアテスト

ユニットテストを実行します：
```bash
python3 -m pytest tests/ -v
```

### 回答精度の評価

SearXNG と Ollama が起動している環境で、音声を使わずにテキスト入出力だけの評価ハーネスを実行できます。基本コマンドは次のとおりです：

```bash
python3 scripts/eval.py
```

主なオプション：

- `--list`: 評価ケースの一覧を表示する
- `--only fact-01,follow-01`: 指定したケースだけ実行する
- `--category followup`: 指定カテゴリだけ実行する
- `--repeat 3`: 各ケースを複数回実行する
- `--no-search`: SearXNG検索を通さずLLM単体で評価する
- `--ask "質問"`: 評価ケースを使わず、任意の質問をその場で確認する。複数指定すると履歴をつないだ会話になる
- `--baseline reports/eval_before.json`: ベースラインとの差分を表示する
- `--markdown reports/eval.md`: Markdownサマリーも保存する

任意の質問を1件だけ確認する例：

```bash
python3 scripts/eval.py --ask "視力を良くする方法は？"
```

複数の質問を続けて確認し、履歴の効き方を見る例：

```bash
python3 scripts/eval.py --ask "富士山の高さは？" --ask "その山はどこにある？"
```

`--ask` には期待値がないため、通常の評価ケースのような PASS/FAIL 判定は出ません。回答を確認するだけの実行ではレポートJSONを保存せず、保存したい場合だけ `--out reports/my_question.json` のように指定します。通常の評価ケースのJSONレポートは既定で `reports/` に保存されます。自動採点は、アルファベット混入、文数、キーワード、再復唱一致など機械判定できる項目だけを対象とします。回答本文の自然さや事実性など、自動判定できない品質はレポートを人が読んで判断してください。

### ハードウェアの物理動作確認

既定の配線は次のとおりです。ボタンは GPIO と GND の間に接続します（内蔵プルアップを使用）。

| 部品 | GPIO | 設定項目 |
| --- | --- | --- |
| ステータスLED | 23 | `STATUS_LED_PIN` |
| プッシュ・トゥ・トークのボタン | 24 | `PTT_BUTTON_PIN` |

Raspberry Piに接続したLEDやボタンが物理的に正しく配線されているか確認するためのスクリプトです。ボタンを押している間LEDが点灯します：
```bash
python3 scripts/button_led_test.py
```

## ディレクトリ構成

```
smart-speaker/
├── AGENTS.md               # AIエージェント向け開発ルール
├── project_context.md      # プロジェクト要件
├── future_extensions.md    # 将来の機能拡張バックログ
├── README.md               # 本ファイル（セットアップ・実行方法）
├── pyproject.toml          # 依存ライブラリの定義
├── .gitignore              # Git除外設定
├── .env.example            # 環境変数テンプレート
├── deploy/                 # ラズパイ常駐運用（systemd）
│   ├── install.sh          # systemdユニットの生成・配置・有効化
│   ├── uninstall.sh        # ユニットの停止・削除
│   ├── searxng.service     # SearXNGコンテナ用ユニットのテンプレート
│   ├── smart-speaker.service  # src.main 常駐用ユニットのテンプレート
│   └── searxng/
│       └── settings.yml    # SearXNGのJSON API有効化設定
├── models/
│   └── reranker/           # オフラインキャッシュ用Rerankingモデル（.gitignore対象）
├── conftest.py             # src/ を sys.path に追加（テスト用）
├── src/
│   ├── __init__.py
│   ├── main.py             # オーケストレーター
│   ├── listener.py         # Whisper音声認識
│   ├── retriever.py        # SearXNG Web検索
│   ├── composer.py         # RAGプロンプト構成
│   ├── query_prep.py       # 検索要否判定・クエリ書き換え
│   ├── text_turn.py        # テキスト入出力の1ターン（main / eval 共用）
│   ├── speech_normalize.py # 発話前の単位・略語の日本語読み化
│   ├── brain.py            # Ollama AI生成
│   ├── speaker.py          # Piper-Plus TTS
│   ├── status_led.py       # GPIO ステータスLED制御
│   ├── push_to_talk.py     # GPIOボタンによるプッシュ・トゥ・トーク
│   ├── conversation_history.py  # マルチターン対話の履歴管理
│   ├── config.py           # 環境変数の一元管理・URL検証
│   ├── http_client.py      # 共通HTTPクライアント
│   ├── exceptions.py       # ドメイン固有の例外
│   └── audio_utils.py      # 音声・ログ共通ユーティリティ
├── scripts/                # 手動実行スクリプト（pytest対象外）
│   ├── button_led_test.py  # LED・ボタンの物理配線確認用
│   ├── eval.py             # 回答精度の手動評価ハーネス
│   └── eval_cases.json     # 評価ケース定義
└── tests/
    ├── __init__.py
    ├── conftest.py         # pytest共通フィクスチャ
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
    ├── test_logging_policy.py  # src/ に print() が無いことを検証
    └── test_eval_harness.py    # 評価ハーネス純粋関数のテスト
```

## 注意点

- **speaker.py**: 本家Piper（espeak-ng依存）は日本語のアクセント解析が未対応のため、必ず `piper-tts-plus` を使用すること
- **retriever.py（Reranking）**: 標準構成として既定で有効。依存ライブラリ未導入時や処理に失敗した場合は自動的にスキップされ、検索結果をそのまま使用する
- **retriever.py（本文取得・二段Rerank）**: `FETCH_PAGE_ENABLED=true`（既定）で、Rerank上位URLを並列取得し`trafilatura`で本文抽出、パッセージ単位で再度Rerankする。パッセージ分割は段落→文境界（句点・`！？!?`・改行）を優先し、境界のない長文は`PASSAGE_CHARS`字ずつ強制分割することで1パッセージが上限を超えないようにする。取得・抽出に失敗した件は元のスニペットへ戻り、検索処理自体は継続する。二段Rerank後は`title`+`content`の合計文字数が`CONTEXT_CHAR_BUDGET`を超えないよう関連度の低いパッセージから間引く（1件も無くなることはない）。`FETCH_PAGE_ENABLED=false`で現行のスニペット経路に戻せる
- **status_led.py**: `gpiozero` は Raspberry Pi 上でのみ動作するため、非Pi環境では自動的に無効化される
- **composer.py**: 本番経路は `compose_messages()`。日本語限定と、質問タイプに合わせた文数（事実は1〜3文、説明は3〜5文、結論先出し）は system メッセージに集約する。検索結果は質問に関係するものだけを事実として扱い、無関係なら使わず、使う場合は推測で補わない。発話に「検索結果」とは言わない。`compose_prompt()` は切り戻し用に残す
- **query_prep.py**: 最終回答の前に、検索要否と検索クエリ書き換えを1回の補助LLM呼び出しで行う。失敗時は元の質問で検索する。結果は発話しない
- **speech_normalize.py**: 単位や略語の英字を、Piper 向けの日本語読みに機械的に置換する。生成後の LLM 推敲ではない
- **conversation_history.py**: 直近 `CONVERSATION_MAX_TURNS` 回分の問いと答えを保持する。`/api/chat` 経路では `as_messages()` で user / assistant の交互メッセージを渡す。「もう一回言って」などの再復唱コマンドは、検索・生成を行わず直前の回答をそのまま発話する
- **push_to_talk.py**: GPIOボタンが利用できる環境では「押している間だけ録音」するプッシュ・トゥ・トークで動作する。ボタンが無い非Pi環境（`gpiozero` 不在）では自動的に無効化され、`RECORD_SECONDS` の固定秒数録音にフォールバックする。`PTT_MIN_RECORD_SECONDS` / `PTT_MAX_RECORD_SECONDS` で最小・最大録音時間を制御する
- **テスト実行時**: 外部API（SearXNG/Ollama）にリクエストを飛ばさず、`pytest-mock` でモック化すること
