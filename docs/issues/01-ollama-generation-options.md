# feat: brain.py に Ollama の生成パラメータ（options）と system プロンプトを追加

> このファイルは GitHub issue の草案です。内容を確定させたうえで issue として起票してください。
> 出典: `docs/現状分析と改善案_実装順序比較.html`「4案を統合した場合の実装順序」順序1

## 背景

Fable / Fusion / Kimi / Ultra の4案すべてが**実施順序の1位**に置いた施策で、依存追加なし・変更小・効果最大という評価も一致している。

現状 `src/brain.py:35-39` は `/api/generate` に `model` / `prompt` / `stream` の3つしか送っておらず、`options` が未指定。

```python
json_body={
    'model': self.ollama_model,
    'prompt': prompt,
    'stream': False
},
```

このため、コンテキスト窓・生成長・温度・繰り返し抑制がすべて Ollama のサーバ既定に委ねられている。4案が共通して指摘している影響は次の2点。

- **コンテキスト窓（`num_ctx`）が既定のまま**のため、検索結果＋会話履歴＋指示文でプロンプトが窓を超えると、Ollama が**古いトークン（＝プロンプト冒頭）から黙って捨てる**。捨てられるのは日本語限定の指示文と会話履歴なので、「会話コンテキストが保持されない」症状として現れる。
- **`temperature` が既定のまま**で、検索結果を事実として扱う用途に対して高すぎる可能性がある。

## スコープ

- 本 issue では **`/api/generate` のまま**とする。`/api/chat` への移行と履歴の messages 化は別 issue（順序3）で行う。効果を段階的に切り分けるため。
- `src/brain.py:27` の `prompt[:10000]` は本 issue では触らない。順序3のトークン予算管理で置き換える。

## 変更内容

### 1. `src/config.py` に生成パラメータを追加

`os.getenv()` は config.py に集約するルールに従い、すべてここで定義する。

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

### 3. `.env.example` を更新

`# Ollama Configuration` セクションに上記の環境変数を追記する。

### 4. `tests/test_brain.py` にテストを追加

- `options` の各キーが期待値で送信されること（`requests.post` のモックに渡った `json` を検証）
- `system` / `keep_alive` が送信されること
- 環境変数で値を上書きできること

外部サービスには実アクセスしない（`pytest-mock` でモック化）。

### 5. 実機での既定値確認（実装とは別作業）

4案の間で Ollama の既定 `num_ctx` の認識が **2048 と 4096 に割れている**ため、実機で確認して記録する。

```bash
curl -s http://localhost:11434/api/show -d '{"name":"qwen2.5:3b"}' | python3 -m json.tool
ollama ps   # CONTEXT 列で実際の割当を確認
```

## 決めるべき数値

| 設定 | Fable | Fusion | Kimi | Ultra | 状況 | 推奨 |
| --- | --- | --- | --- | --- | --- | --- |
| `num_ctx` | 8192 | 8192 | **4096** | 8192 | **不一致** | 8192 |
| `num_predict` | 512 | 512 | 512 | 512 | 一致 | 512 |
| `temperature` | 0.2〜0.3 | 0.3 | 0.3 | **0.7** | **不一致** | 0.3 |
| `repeat_penalty` | 指定を推奨（値の記載なし） | 1.1 | 1.1 | 1.1 | 実質一致 | 1.1 |
| `top_p` | — | 0.9 | — | — | Fusion のみ | 今回は入れない |
| `stop` | — | `["質問：", "検索結果：", "\nユーザー"]` | — | — | Fusion のみ | 今回は入れない |
| `keep_alive` | — | — | 30m | — | Kimi のみ | 入れる |

### 判断材料

- **`num_ctx` 8192 か 4096 か**: Pi 5（8GB）ではモデル本体と KV キャッシュがメモリを取り合う。qwen2.5:3b Q4_K_M（約2GB）＋ 8192 トークン分の KV キャッシュで、増加分は数百MB規模の見込み。順序6の本文取得（1,200〜2,000字を追加）まで見据えるなら 8192 が必要になる。Kimi が 4096 を採るのは他の施策（本文取得を上位1〜2件・800〜1200字に抑える）とセットの前提。
- **`temperature` 0.3 か 0.7 か**: 「検索結果を絶対に事実として扱う」用途では低いほうが忠実になる。0.7 は Ultra のみ。
- **`top_p` / `stop`**: 順序3で `/api/chat` に移行すると ChatML が適用され、`質問：` などの自己生成は起きにくくなるため `stop` の必要性は下がる。パラメータを一度に増やすと効果の切り分けが難しくなるので、まず4つに絞るのを推奨。
- **`keep_alive`**: 順序5（クエリ書き換え）で LLM を複数回呼ぶようになると、モデルのアンロード／再ロード待ちが顕在化する。先に入れておいて損はない。

## 注意点

- **`num_predict: 512` は暴走防止の上限であり、回答を長くする施策ではない。** Kimi は「Ollama の `num_predict` 既定は -1（無制限）で、ドキュメントの 128 は誤記」と指摘しており、他3案の「未指定だから短い」という説明と対立している。**「回答が短すぎる」への本命の対策は、順序2（別 issue）のプロンプト文量指示。** 本 issue だけで回答長が改善しなくても想定内。
- `system` に入れる日本語限定・アルファベット禁止の指示は、AGENTS.md の制約（TTS 安定性のため日本語のみ）を維持するもの。ここで確定した system 文は、順序3の `/api/chat` の system メッセージにそのまま流用する。
- `/api/generate` は `raw` 未指定だとモデルのチャットテンプレートが適用され、`system` を渡さないと Qwen 既定の英語システム文言が暗黙に挿入される（Kimi の指摘）。`system` を明示するとこれを上書きできる。

## 完了条件

- [ ] `python3 -m pytest tests/ -v` が全パス
- [ ] `.env.example` を更新
- [ ] 実機で `ollama ps` の CONTEXT 列が指定した `num_ctx` になっていることを確認
- [ ] feature ブランチ ＋ PR 経由でマージ（`main` への直接 push 禁止）
