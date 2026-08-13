# 回答精度改善 フェーズ1〜3: 生成パラメータ / プロンプト強化 / `/api/chat` 移行

> このファイルは GitHub issue の草案です。内容を確認のうえ issue として起票してください。
> 出典: `docs/現状分析と改善案_実装順序比較.html`「4案を統合した場合の実装順序」順序1〜3

## 背景

症状は「回答が短すぎる」「会話コンテキストが保持されない」の2点。Fable / Fusion / Kimi / Ultra の4案を比較した結果、いずれも**依存追加なしで着手できる3施策**を最優先に置いており、順序もほぼ一致していた。本 issue ではその3件をフェーズ1〜3として扱う。

各フェーズは**独立した PR に分ける**（AGENTS.md「1 PR ＝ 1目的」）。効果を切り分けるため、フェーズ1と2をマージして実機で再評価してからフェーズ3に着手する。

| フェーズ | 内容 | 主に効く症状 | 依存追加 | 追加遅延 |
| --- | --- | --- | --- | --- |
| 1 | `brain.py` に `options` と `system` を追加 | 文脈の消失 | なし | なし |
| 2 | `composer.py` のプロンプト強化 | 回答が短い | なし | なし |
| 3 | `/api/chat` 移行・履歴の messages 化・トークン予算管理 | 文脈の消失 | なし | なし |

## 確定した設定値

4案で値が割れていた項目はすべて決定済み。

| 設定 | 現在 | 決定値 | 備考 |
| --- | --- | --- | --- |
| `OLLAMA_NUM_CTX` | 未指定 | **8192** | フェーズ1 |
| `OLLAMA_NUM_PREDICT` | 未指定 | **512** | 4案一致。暴走防止の上限 |
| `OLLAMA_TEMPERATURE` | 未指定 | **0.3** | 検索結果への忠実性を優先 |
| `OLLAMA_REPEAT_PENALTY` | 未指定 | **1.1** | 4案一致 |
| `OLLAMA_KEEP_ALIVE` | 未指定 | **30m** | 30分。フェーズ5以降の複数回呼び出しに備える |
| 回答の文数 | 指示なし | **3〜5文**（結論→補足） | フェーズ2 |
| 無関係な検索結果の扱い | 「絶対に事実」のみ | **使わずに会話履歴と知識で答える／使う場合は推測で補わない** | フェーズ2 |
| `CONVERSATION_MAX_TURNS` | 3 | **5** | フェーズ3 |
| `CONVERSATION_ANSWER_CLIP` | 200 | **400** | フェーズ3 |
| `CONTEXT_CHAR_BUDGET` | なし | **2000** | 検索コンテキストの文字予算。フェーズ3 |
| `CHAR_TO_TOKEN_RATIO` | なし | **1.0** | 日本語は1文字≒1トークンとして保守的に見積もる |
| `OLLAMA_API_URL` | `.../api/generate` | **`.../api/chat` に差し替え** | フェーズ3。変数は増やさない |

**採用を見送るもの**: `top_p`（Fusion のみ提案）、`stop`（同左。`/api/chat` 移行後は自己生成が起きにくく必要性が下がる）、出典番号 `[1]` の付与（Piper が「1によると」を読み上げるリスク）。いずれもフェーズ3完了後に必要なら再検討する。

---

# フェーズ1: `brain.py` に生成パラメータと system プロンプトを追加

## 現状

`src/brain.py:35-39` は `/api/generate` に `model` / `prompt` / `stream` の3つしか送っておらず、`options` が未指定。

```python
json_body={
    'model': self.ollama_model,
    'prompt': prompt,
    'stream': False
},
```

コンテキスト窓・生成長・温度がすべてサーバ既定に委ねられている。特に `num_ctx` が既定のままだと、検索結果＋会話履歴＋指示文でプロンプトが窓を超えたときに Ollama が**古いトークン（＝プロンプト冒頭）から黙って捨てる**。捨てられるのは日本語限定の指示文と会話履歴なので、「会話コンテキストが保持されない」症状として現れる。

本フェーズでは **`/api/generate` のまま**とする（`/api/chat` 移行はフェーズ3）。`src/brain.py:27` の `prompt[:10000]` も本フェーズでは触らない（フェーズ3で置き換え）。

## 変更内容

### 1. `src/config.py` に生成パラメータを追加

`os.getenv()` は config.py に集約するルールに従う。

```python
# Brain (Ollama) generation settings
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
OLLAMA_REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.1"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_SYSTEM_PROMPT = os.getenv(
    "OLLAMA_SYSTEM_PROMPT",
    "あなたは日本語専用の音声アシスタントです。"
    "回答はすべて日本語のみで行い、アルファベット（英語の単語や文）を含めてはいけません。"
    "必要であればカタカナや日本語表現に翻訳して出力してください。",
)
```

### 2. `src/brain.py` の `json_body` を拡張

```python
json_body={
    'model': self.ollama_model,
    'prompt': prompt,
    'stream': False,
    'system': OLLAMA_SYSTEM_PROMPT,
    'keep_alive': OLLAMA_KEEP_ALIVE,
    'options': {
        'num_ctx': OLLAMA_NUM_CTX,
        'num_predict': OLLAMA_NUM_PREDICT,
        'temperature': OLLAMA_TEMPERATURE,
        'repeat_penalty': OLLAMA_REPEAT_PENALTY,
    },
},
```

### 3. `.env.example` の `# Ollama Configuration` セクションを更新

```bash
OLLAMA_API_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_NUM_CTX=8192
OLLAMA_NUM_PREDICT=512
OLLAMA_TEMPERATURE=0.3
OLLAMA_REPEAT_PENALTY=1.1
# モデルをメモリに保持する時間。期間文字列（30m / 24h）または秒数（1800）で指定する
OLLAMA_KEEP_ALIVE=30m
```

### 4. `tests/test_brain.py` にテストを追加

- `options` の各キーが期待値で送信されること（`requests.post` のモックに渡った `json` を検証）
- `system` / `keep_alive` が送信されること
- 環境変数で値を上書きできること

外部サービスには実アクセスせず、`pytest-mock` でモック化する。

### 5. 実機での既定値確認（実装とは別作業）

4案の間で Ollama の既定 `num_ctx` の認識が **2048 と 4096 に割れている**ため、実機で確認して issue にコメントで記録する。

```bash
curl -s http://localhost:11434/api/show -d '{"name":"qwen2.5:3b"}' | python3 -m json.tool
ollama ps   # CONTEXT 列で実際の割当が 8192 になっているか確認
```

## 注意点

- **`num_predict: 512` は暴走防止の上限であり、回答を長くする施策ではない。** Kimi は「Ollama の `num_predict` 既定は -1（無制限）で、ドキュメントの128は誤記」と指摘しており、他3案の「未指定だから短い」という説明と対立している。**「回答が短すぎる」への本命の対策はフェーズ2のプロンプト文量指示**であり、本フェーズだけで回答長が改善しなくても想定内。
- `keep_alive` の `30m` は**30分**（ミリ秒ではない）。Ollama は期間文字列（`10m` / `24h`）または**秒数**の数値を受け付け、既定は `5m`。負値で常駐、`0` で応答後すぐアンロード。`ollama ps` の UNTIL 列で実際の保持期限を確認できる。
- `/api/generate` は `raw` 未指定だとモデルのチャットテンプレートが適用され、`system` を渡さないと Qwen 既定の英語システム文言が暗黙に挿入される。`system` を明示することでこれを上書きする。
- `num_ctx` 8192 による KV キャッシュの増加は数百MB規模の見込み。qwen2.5:3b Q4_K_M（約2GB）と合わせても Pi 5（8GB）に収まるが、実機のメモリ使用量を確認すること。

---

# フェーズ2: `composer.py` のプロンプト強化

## 現状

「回答が短すぎる」への**本命の対策**。4案とも「小型モデルは文量の指示がないと最短で打ち切る」と分析している。

現行プロンプト（`src/composer.py:29-37`）は「日本語のみ」「アルファベット禁止」しか要求しておらず、**回答の分量・構成の指示が一切ない**。末尾が `回答：` の穴埋め形式なので、3B クラスのモデルは最小限で埋めて終了しがち。

もう1点、`composer.py:29` の「**絶対に事実として扱い**」という強い縛りが、フォローアップ質問時に問題を起こす。指示語だけの発話（「それはどういうこと？」）がそのまま検索クエリになって無関係な結果が返っても、それが「絶対的事実」として注入され、モデルが会話履歴より検索結果を優先してしまう。検索クエリ側の根本対策は別施策（統合順序5のクエリ書き換え）だが、プロンプト側でも緩和する。

## 変更内容

### 1. 検索結果ありの分岐（`composer.py:29-37`）

```python
prompt = f"""以下の検索結果のうち、質問に関係するものを『絶対に事実』として扱い、ユーザーの質問に日本語のみで答えなさい。
回答にはアルファベット（英語の単語や文）を含めず、必要であればカタカナや日本語表現に翻訳して出力してください。
回答は3〜5文で、結論を先に述べてから補足を加える形で、具体的に説明しなさい。
「それ」「さっきの」などの指示語は、これまでの会話を参照して解釈しなさい。
検索結果が質問と無関係な場合は、検索結果を使わず、これまでの会話とあなたの知識で答えなさい。
検索結果を使って答える場合は、検索結果に書かれていないことを推測で補ってはいけません。

{history_block}検索結果：
{context}

質問：{query}

回答："""
```

1行目を「以下の検索結果を」から「**以下の検索結果のうち、質問に関係するものを**」に変更している点に注意。これがないと「絶対に事実として扱え」と「無関係なら使うな」が正面から矛盾する。

### 2. 検索結果なしの分岐（`composer.py:39`）

検索失敗は非致命でこの経路を通るため、同じ文量指示を効かせる。質問は末尾に置いたままにする。

```python
prompt = f"""日本語のみで、アルファベット（英語の単語や文）を含めずに答えなさい。
回答は3〜5文で、結論を先に述べてから補足を加える形で、具体的に説明しなさい。
「それ」「さっきの」などの指示語は、これまでの会話を参照して解釈しなさい。

{history_block}質問：{query}

回答："""
```

### 3. `tests/test_composer.py` を更新

- 追加した指示文がプロンプトに含まれること
- 履歴あり／なし × 検索結果あり／なしの4通りで構造が保たれること
- **日本語限定・アルファベット禁止の文言が維持されていること**（AGENTS.md の制約が退行していないことの回帰テスト）

## 現状ですでに満たされている点（変更不要）

- **質問はすでにプロンプト末尾**（`回答：` の直前）にある。Fusion の「質問を最後に置いて recency bias を活用する」は対応済み。
- 日本語限定・アルファベット禁止の指示は AGENTS.md の必須制約。**削除・弱体化しないこと。**

## 確認方法

回答長の改善はフェーズ1の効果と切り分ける必要があるため、**フェーズ1をマージした状態を基準に**前後比較する。3〜5文は概ね150〜250字で、Piper の読み上げは20〜40秒程度になる見込み。実機で長すぎるようなら文数を下げる。

---

# フェーズ3: `/api/chat` 移行・履歴の messages 化・トークン予算管理

## 現状

「会話コンテキストが保持されない」への構造的な対策のうち、生成側の対策。

現状は `/api/generate` に**1本の文字列**を投げており、日本語限定の指示・会話履歴・検索結果・質問がすべて同じ平文に混在している。会話履歴は間接話法の要約文字列（`conversation_history.py:75`）。

```python
lines.append(f"ユーザーは「{query}」と質問し、「{self._clip(answer)}」と回答された。")
```

4案が共通して指摘している問題は次のとおり。

- Qwen2.5 は**チャット調整済みモデル**で、指示追従の多くをチャットテンプレート（ChatML）に依存している。role が分離されていない平文だと指示無視・一文回答が起きやすい。
- 履歴の平文が検索結果ブロックの直上にあり、**役割の境界がない**ためモデルが両者を混同する。
- 履歴はプロンプト**前方**にあるので、コンテキスト窓を超えたときに真っ先に切り捨てられる。
- `brain.py:27` の `prompt[:10000]` は「1万**文字**」での単純切り捨てで `num_ctx` とは無関係な数字。しかも先頭を残して末尾（＝質問）を捨てる方向なので、ガードとして機能していない。

Fable / Fusion / Ultra はこれをフェーズ1に含めているが、**Kimi のみ後ろ倒し**を主張（まず低リスクな施策で効果を見る段階論）。本 issue ではフェーズ1・2の効果測定後に着手する位置づけとした。

## 変更内容

### 1. `src/brain.py`: `/api/chat` へ移行

- POST 先を `/api/chat` に変更（`OLLAMA_API_URL` の値を差し替える）
- `prompt` 文字列ではなく `messages` 配列を受け取る
- レスポンスの取り出しを `data['response']` から `data['message']['content']` に変更
- `options` / `keep_alive` はフェーズ1で追加したものをそのまま使う

### 2. `src/composer.py`: system / user の分離

`compose_prompt()` に加えて `compose_messages()` を追加する。

```python
[
    {"role": "system", "content": SYSTEM_PROMPT},
    *history_messages,                    # user / assistant の交互
    {"role": "user",   "content": "検索結果：\n...\n\n質問：..."},
]
```

フェーズ2で確定した指示は system メッセージに集約する。

```
あなたは日本語専用の音声アシスタントです。
回答はすべて日本語のみで行い、アルファベット（英語の単語や文）を含めてはいけません。必要であればカタカナや日本語表現に翻訳してください。
回答は3〜5文で、結論を先に述べてから補足を加える形で、具体的に説明してください。
「それ」「さっきの」などの指示語は、これまでの会話を参照して解釈してください。
検索結果が質問と無関係な場合は、検索結果を使わず、これまでの会話とあなたの知識で答えてください。
検索結果を使って答える場合は、検索結果に書かれていないことを推測で補ってはいけません。
```

### 3. `src/conversation_history.py`: `as_messages()` を追加

```python
def as_messages(self):
    """Render retained turns as role-tagged chat messages."""
    messages = []
    for query, answer in self._turns:
        messages.append({"role": "user", "content": query})
        messages.append({"role": "assistant", "content": self._clip(answer)})
    return messages
```

既存の `as_condensed_context()` は削除せず残す（`compose_prompt()` 経路の後方互換のため）。

### 4. `src/main.py`: 受け渡しを messages 経路に変更

`history.as_condensed_context()` → `history.as_messages()`、`composer.compose_prompt()` → `composer.compose_messages()`。エラー方針（`SearchError` は非致命、`GenerationError` は致命）は変更しない。

`compose_prompt()` / `as_condensed_context()` / 旧 `generate_response(prompt)` は関数としては残し、`main.py` だけが `/api/chat` 経路を使う形にする。問題が出たときに `main.py` の数行で切り戻せるようにするため。

### 5. トークン予算管理（`prompt[:10000]` の置き換え）

`brain.py:27` の一括切り捨てを廃止し、以下に置き換える。

```python
CONTEXT_CHAR_BUDGET = int(os.getenv("CONTEXT_CHAR_BUDGET", "2000"))
CHAR_TO_TOKEN_RATIO = float(os.getenv("CHAR_TO_TOKEN_RATIO", "1.0"))
```

1. 検索コンテキストを `CONTEXT_CHAR_BUDGET`（2000字）に収める。溢れる場合は**検索結果の単位で後ろから落とす**
2. messages 全体の概算トークン（総文字数 × `CHAR_TO_TOKEN_RATIO`）が `OLLAMA_NUM_CTX - OLLAMA_NUM_PREDICT`（8192 − 512 ＝ 7680）を超える場合、**古い履歴ターンから落とす**
3. **system メッセージと最後の user メッセージ（検索結果＋質問）は落とさない**

換算係数を 1.0（1文字＝1トークン）にしているのは、溢れさせないことが目的なので過小評価より過大評価が安全なため。現時点の想定使用量は system 約200字＋履歴5ターン（各質問＋400字）＋検索コンテキスト最大2000字で、合計およそ4,700トークン相当。7,680の予算内に収まるため、この予算が実際に効く場面は本フェーズではまだ少ない。統合順序6の本文取得を入れた段階で実測して見直す。

### 6. `.env.example` と README の更新

```bash
OLLAMA_API_URL=http://localhost:11434/api/chat
CONTEXT_CHAR_BUDGET=2000
CHAR_TO_TOKEN_RATIO=1.0

CONVERSATION_MAX_TURNS=5
CONVERSATION_ANSWER_CLIP=400
```

`OLLAMA_API_URL` は**既存の `.env` を持っている環境で書き換えが必要**になる。README に移行手順を明記すること。

### 7. テストの更新

- `tests/test_brain.py`: モックのレスポンスを `{'response': ...}` から `{'message': {'content': ...}}` に変更。`messages` が正しい構造で送信されることを検証
- `tests/test_conversation_history.py`: `as_messages()` の role 順序と clip 適用のテストを追加
- `tests/test_composer.py`: `compose_messages()` のテストを追加
- `tests/test_main.py`: messages 経路の呼び出しに追随
- トークン予算管理の境界テスト（予算超過時に検索コンテキストから削られ、質問と直近履歴が残ること）

## 確認方法

実機でフォローアップ質問（「それはどういうこと？」「もっと詳しく」）を投げ、直前の話題を踏まえた回答になるかを確認する。なお、指示語がそのまま検索クエリになる問題自体は本フェーズでは解消しない（統合順序5のクエリ書き換えが担当）。本フェーズで改善するのは、**履歴が生成側に正しく届くこと**まで。

---

## 残っている判断

- **直近1ターンだけ `CONVERSATION_ANSWER_CLIP` の対象外にするか**（Fusion のみの提案）。「もっと詳しく」のように直前の回答そのものを参照する発話に効く。まずは全ターン一律400字で実装し、必要なら追加する。

## 完了条件

- [ ] フェーズごとに独立した PR（計3本）に分割されている
- [ ] `python3 -m pytest tests/ -v` が各 PR で全パス（既存テストが退行していないこと）
- [ ] `.env.example` が更新されている
- [ ] 日本語限定・アルファベット禁止の指示が維持され、テストで担保されている
- [ ] フェーズ1で `ollama ps` の CONTEXT 列が 8192 になっていることを実機確認
- [ ] フェーズ2の完了時点で回答長を前後比較
- [ ] フェーズ3で `prompt[:10000]` が削除され、`num_ctx` 由来の予算管理に置き換わっている
- [ ] フェーズ3で `OLLAMA_API_URL` の値の変更が README に明記されている
- [ ] すべて feature ブランチ ＋ PR 経由（`main` への直接 push 禁止）
