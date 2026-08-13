# feat: /api/chat への移行と会話履歴の messages 化、トークン予算管理の導入

> このファイルは GitHub issue の草案です。内容を確定させたうえで issue として起票してください。
> 出典: `docs/現状分析と改善案_実装順序比較.html`「4案を統合した場合の実装順序」順序3

## 背景

「会話コンテキストが保持されない」への構造的な対策のうち、生成側の対策。

現状は `/api/generate` に**1本の文字列**を投げており、日本語限定の指示・会話履歴・検索結果・質問がすべて同じ平文に混在している。会話履歴は間接話法の要約文字列で表現されている（`conversation_history.py:75`）。

```python
lines.append(f"ユーザーは「{query}」と質問し、「{self._clip(answer)}」と回答された。")
```

4案が共通して指摘している問題は次のとおり。

- Qwen2.5 は**チャット調整済みモデル**であり、指示追従の多くをチャットテンプレート（ChatML）に依存している。role が分離されていない平文だと、指示無視・一文回答が起きやすい。
- 履歴の平文が検索結果ブロックの直上にあり、**役割の境界がない**ため、モデルが履歴と検索結果を混同する。
- 履歴はプロンプト**前方**に置かれているので、コンテキスト窓を超えたときに**真っ先に切り捨てられる**。
- `brain.py:27` の `prompt[:10000]` は「1万**文字**」での単純切り捨てで、`num_ctx` とは無関係な数字。しかも先頭を残して末尾（＝質問）を捨てる方向なので、ガードとして機能していない。

順序としては Fable / Fusion / Ultra が1番目に含めているのに対し、**Kimi のみ5番目に後ろ倒し**（まず低リスクな順序1・2で効果を見る段階論）。統合案では、順序1・2の効果測定が済んだ後に着手する位置づけとした。

## スコープ

順序1（`options` 追加）と順序2（プロンプト強化）がマージ済みであることを前提とする。本 issue で変更するのは以下。

- `/api/generate` → `/api/chat` への移行
- 会話履歴の messages 化
- 履歴の保持量（turns / clip）の拡張
- `prompt[:10000]` のトークン予算管理への置き換え

**含めないもの**: クエリ書き換え（順序5）、本文取得（順序6）、ストリーミング（順序7）。

## 変更内容

### 1. `src/brain.py`: `/api/chat` へ移行

- POST 先を `/api/chat` に変更
- `prompt` 文字列ではなく `messages` 配列を受け取る
- レスポンスの取り出しを `data['response']` から `data['message']['content']` に変更
- `options` / `keep_alive` は順序1で追加したものをそのまま使う

### 2. `src/composer.py`: system / user の分離

`compose_prompt()` に加えて `compose_messages()` を追加し、次の構造を返す。

```python
[
    {"role": "system",    "content": <順序1・2で確定した指示文>},
    *history_messages,                       # user / assistant の交互
    {"role": "user",      "content": "検索結果：\n...\n\n質問：..."},
]
```

順序2でプロンプトに入れた文量・構成・指示語解釈・無関係な検索結果の扱いの指示は、**system メッセージに集約**する。検索結果と今回の質問は最後の user メッセージに置く。

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

`history.as_condensed_context()` → `history.as_messages()`、`composer.compose_prompt()` → `composer.compose_messages()`。エラー方針（`GenerationError` は致命、`SearchError` は非致命）は変更しない。

### 5. トークン予算管理（`prompt[:10000]` の置き換え）

`brain.py:27` の一括切り捨てを廃止し、以下の方針に置き換える（Fusion 案）。

- 入力に使える予算を `num_ctx - num_predict` から逆算する
- 予算超過時は**検索コンテキストから削る**。**会話履歴と今回の質問は死守**する
- それでも足りない場合のみ、古い履歴ターンから落とす
- トークン数は tokenizer を持ち込まず、文字数からの概算で十分（換算係数は下記で決定）

### 6. `.env.example` と設定値の更新

### 7. テストの更新

- `tests/test_brain.py`: モックのレスポンスを `{'response': ...}` から `{'message': {'content': ...}}` に変更。`messages` が正しい構造で送信されることを検証
- `tests/test_conversation_history.py`: `as_messages()` の role 順序・clip 適用のテストを追加
- `tests/test_composer.py`: `compose_messages()` のテストを追加
- `tests/test_main.py`: messages 経路の呼び出しに追随

## 決めるべき数値

| 設定 | 現在値 | Fable | Fusion | Kimi | Ultra | 推奨 |
| --- | --- | --- | --- | --- | --- | --- |
| `CONVERSATION_MAX_TURNS` | 3 | 言及なし | 6 | 言及なし | 5 | **5** |
| `CONVERSATION_ANSWER_CLIP` | 200 | 400〜600 | 500（直近1ターンはクリップなし） | 言及なし | 400 | **400** |
| 直近1ターンをクリップ対象外にするか | — | — | する | — | — | **要判断** |
| 検索コンテキストの文字予算 | なし | — | `CONTEXT_CHAR_BUDGET=1600` | — | — | **2000** |
| 文字→トークンの換算係数 | — | — | 文字数 × 0.7 | — | — | **1.0** |

### 判断材料

- **`CONVERSATION_MAX_TURNS`**: 5 と 6 の差は実効的には小さい。`num_ctx` を 8192 にするなら 6 でも収まるが、古い話題を引きずる副作用もある（Fusion はこれを避けるために話題リセット機能をセットで提案している。本 issue のスコープ外）。
- **`CONVERSATION_ANSWER_CLIP`**: 順序2で回答が3〜5文（150〜250字）になる想定なので、400 あればほぼクリップされずに済む。500 にする実益は小さい。ただし Fusion の「**直近1ターンだけクリップなし**」は、「もっと詳しく」のような直前回答を参照する発話に効くので、採用する価値がある。
- **検索コンテキストの文字予算**: Fusion の 1600 は順序6（本文取得でパッセージを1,200〜1,600字入れる）を前提にした値。本 issue の時点ではまだスニペット3件（合計数百字）しかないため、予算が実際に効く場面はほぼない。2000 程度を置いておき、順序6で実測して見直すのが現実的。
- **換算係数**: Fusion は文字数×0.7 でトークンを概算する想定。日本語は Qwen 系のトークナイザでも概ね1文字あたり0.7〜1.0トークン程度になるため、**切り詰めのガードとしては保守的な 1.0（1文字＝1トークン）で見積もるほうが安全**。溢れさせないことが目的なので、過小評価より過大評価のほうが望ましい。

## 決めるべき設計判断

### 1. 環境変数名をどうするか

Fusion は `OLLAMA_CHAT_API_URL` を新設する案。一方、既存の `OLLAMA_API_URL`（現在 `.../api/generate`）の**値だけを `.../api/chat` に差し替える**方法もある。

**推奨: `OLLAMA_API_URL` の値を差し替える。** 同じ役割の設定を2つ持つと、AGENTS.md の「設定は config.py で一元管理し重複を避ける」という方針から外れるため。既存の `.env` を持っている環境では書き換えが必要になるので、README と `.env.example` に移行手順を明記する。

### 2. `/api/generate` 経路を残すか

`compose_prompt()` / `as_condensed_context()` / `generate_response(prompt)` を残してフォールバック可能にするか、`/api/chat` 一本にするか。

**推奨: 関数は残すが、`main.py` は `/api/chat` 経路のみを使う。** 既存テストを大きく壊さずに済み、問題が出たときに `main.py` の数行で切り戻せる。

## 完了条件

- [ ] `python3 -m pytest tests/ -v` が全パス（既存97件が退行していないこと）
- [ ] `.env.example` を更新し、`OLLAMA_API_URL` の値の変更を README に明記
- [ ] 実機でフォローアップ質問（「それはどういうこと？」等）の追従が改善したか確認
- [ ] `prompt[:10000]` が削除され、`num_ctx` 由来の予算管理に置き換わっていること
- [ ] feature ブランチ ＋ PR 経由でマージ
