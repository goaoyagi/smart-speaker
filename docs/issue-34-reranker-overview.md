# Issue #34: Optimum ONNX 日本語 Reranker 統合 — 概要

> 参照: [GitHub Issue #34](https://github.com/goaoyagi/smart-speaker/issues/34)

---

## 1. Issue #34 の内容

**タイトル:** Integration of Optimum ONNX Japanese Reranker  
**状態:** Open  
**作成日:** 2026-07-12

Optimum ONNX ベースの日本語 Reranker を、2 段階（Phase 1: 設計ドキュメント → Phase 2: コード実装）で組み込む計画。

### Phase 1: 設計・ドキュメント更新

#### `project_context.md` の変更

- システムアーキテクチャに、`retriever.py` 内の**任意の Re-ranking ステップ**を追加
- 「各コンポーネントの実装注意点」に `retriever.py` の仕様を追記:
  - `optimum` / `transformers` は**動的 import**（未インストール時のクラッシュ防止）
  - ユニットテストでは Reranking を**モック化または無効化**（オフラインルール準拠）

#### `README.md` の変更

- 依存関係に追加: `optimum[onnxruntime] transformers fugashi unidic-lite`
- ディレクトリ構成に `models/reranker/`（オフラインキャッシュ用）を追記
- インターネット制限環境向けの `optimum-cli` によるオフライン export 手順を記載

### Phase 1.5: 依存関係の検証（提案）

実装前に、上記パッケージがローカル環境（Windows / ARM64 / Raspberry Pi など）で問題なくインストールできるか確認する。

### Phase 2: コード実装・検証

Phase 1 完了・承認後に実装。

#### `.env.example` の変更

```bash
# Reranker (Optimum/ONNX) settings
RERANKER_ENABLED=false
RERANKER_MODEL_ID=hotchpotch/japanese-reranker-cross-encoder-xsmall-v1
RERANKER_LOCAL_PATH=./models/reranker
RERANK_TOP_K=3
```

> **補足:** Issue 本文は `RERANK_TOP_K=3` の直後で途切れており、Phase 2 の残り（`config.py` / `retriever.py` の実装、`tests/` の追加など）の記載は GitHub 上には見当たらない。

---

## 2. 機能追加の意図

### 一言で言うと

**「SearXNG が返した検索結果のうち、本当に質問に合うものだけを選んで LLM に渡す」** こと。

### 背景：今の検索フローの限界

現在の `retriever.py` は、SearXNG の結果を**上位 5 件をそのまま** `composer.py` に渡している。

```python
for result in data.get('results', [])[:5]:
    results.append({
        'title': result.get('title', ''),
        'content': result.get('content', ''),
        'url': result.get('url', '')
    })
```

SearXNG の並び順はキーワード一致などに基づくため、**ユーザーの質問に意味的に最も合う結果が上位とは限らない**。

このプロジェクトは「生成前 RAG」でハルシネーションを防ぐ設計なので、**渡すコンテキストの質**が回答の正確さに直結する。関係の薄い検索結果が混ざると、LLM（Qwen2.5:3b）が的外れな情報を「事実」として扱ってしまうリスクがある。

### Reranker が解決すること

**Reranker（再ランキング）** は、クエリと各検索結果のペアをスコアリングし、「この質問に対してどれが本当に役立つか」を再評価する。

```
[現状]
  質問 → SearXNG → 上位5件をそのまま → プロンプト → LLM

[追加後]
  質問 → SearXNG → Reranker で再スコア → 上位K件（例: 3件）→ プロンプト → LLM
```

Issue #34 で選ばれている `hotchpotch/japanese-reranker-cross-encoder-xsmall-v1` は**日本語向け Cross-Encoder**。日本語の音声質問に対して、検索スニペットの関連度をより正確に測れる。

### なぜ Optimum + ONNX なのか

| 選択 | 意図 |
|------|------|
| **Optimum + ONNX Runtime** | PyTorch より軽量・高速に推論でき、Pi でも現実的 |
| **xsmall モデル** | 精度と速度のバランス。Pi のリソース制約を考慮 |
| **ローカルキャッシュ (`models/reranker/`)** | オフライン環境でも動作可能 |
| **`RERANKER_ENABLED=false`（デフォルト）** | 未導入環境でも既存動作を壊さない |

クラウド API に頼らず、端末内で検索結果の品質を上げる — これが「ローカル RAG」の思想に沿った追加。

### プロジェクト全体の目的との関係

`project_context.md` の核心は**「生成前に良い根拠を渡す」**こと。

| コンポーネント | 役割 |
|-------------|------|
| **listener** | 音声を正確にテキスト化 |
| **retriever + reranker** | 質問に合う事実ソースを選ぶ ← **今回の追加** |
| **composer + brain** | 選ばれたソースだけを根拠に回答生成 |
| **speaker** | 日本語のみで発話 |

Reranker は新しい機能というより、**RAG パイプラインの「検索精度」部分を強化する**位置づけ。SearXNG が「候補を集める」役、Reranker が「本当に使う候補を絞る」役になる。

### 設計上の配慮

- **任意機能（opt-in）:** 依存が重い ONNX 系を必須にせず、段階的導入できる
- **動的 import:** `optimum` 未インストールでもクラッシュしない
- **テストでは無効化:** 外部モデル・ネットワークに触れない既存ルールを維持
- **Phase 1 でドキュメント先行:** 実装前に設計を固め、Pi 上での依存問題（Phase 1.5）を先に確認

---

## 3. データフォーマット（SearXNG → Retriever → Reranker）

Reranker は Issue #34 時点では**未実装**。以下は現行コードと Issue #34 の設計に基づく入出力の説明。

### 3.1 SearXNG から返るフォーマット（生 JSON）

`retriever.py` は `GET {SEARXNG_URL}/search?q=...&format=json&language=ja` を呼ぶ。SearXNG は **HTTP レスポンス全体が JSON**。

**リクエスト例:**

```
GET http://localhost:8080/search?q=今日の東京の天気&format=json&language=ja
```

**レスポンス例:**

```json
{
  "query": "今日の東京の天気",
  "number_of_results": 42,
  "results": [
    {
      "url": "https://weather.example.jp/tokyo",
      "title": "東京の天気予報 - 今日・明日",
      "content": "東京都心部は晴れ、最高気温28度。午後から雲が広がる見込みです。",
      "engine": "google",
      "engines": ["google", "bing"],
      "score": 4.5,
      "category": "general",
      "publishedDate": null
    },
    {
      "url": "https://news.example.jp/weather-history",
      "title": "100年前の今日の東京の天気",
      "content": "1926年7月25日の東京は曇りでした。",
      "engine": "duckduckgo",
      "engines": ["duckduckgo"],
      "score": 3.2,
      "category": "general"
    },
    {
      "url": "https://travel.example.jp/tokyo-guide",
      "title": "東京観光ガイド - おすすめスポット",
      "content": "東京タワー、浅草、スカイツリーなど人気の観光地を紹介。",
      "engine": "bing",
      "engines": ["bing"],
      "score": 2.8,
      "category": "general"
    },
    {
      "url": "https://wiki.example.jp/weather-forecast",
      "title": "天気予報とは",
      "content": "天気予報は気象データに基づいて将来の天候を予測するものです。",
      "engine": "wikipedia",
      "engines": ["wikipedia"],
      "score": 2.1,
      "category": "general"
    },
    {
      "url": "https://shop.example.jp/umbrella",
      "title": "雨具セール - 東京店舗一覧",
      "content": "折りたたみ傘が半額。東京各店舗で販売中。",
      "engine": "google",
      "engines": ["google"],
      "score": 1.5,
      "category": "general"
    }
  ],
  "answers": [],
  "suggestions": ["東京 天気 週間", "東京 気温"],
  "corrections": [],
  "infoboxes": [],
  "unresponsive_engines": []
}
```

**ポイント:**

- **`results`** が検索ヒットの配列（メインで使う部分）
- 各要素に `title`, `url`, `content`（スニペット）がある
- SearXNG 自身の **`score`** もあるが、キーワード・エンジン統合の並びであり、質問との意味的一致とは限らない

### 3.2 `retriever.py` が返すフォーマット（現状）

生 JSON から**必要 3 フィールドだけ**抜き出し、**上位 5 件**に切り詰める。

上の SearXNG 例だと、返り値は次のとおり（**SearXNG の並び順のまま**）:

```python
[
  {
    "title": "東京の天気予報 - 今日・明日",
    "content": "東京都心部は晴れ、最高気温28度。午後から雲が広がる見込みです。",
    "url": "https://weather.example.jp/tokyo"
  },
  {
    "title": "100年前の今日の東京の天気",
    "content": "1926年7月25日の東京は曇りでした。",
    "url": "https://news.example.jp/weather-history"
  },
  {
    "title": "東京観光ガイド - おすすめスポット",
    "content": "東京タワー、浅草、スカイツリーなど人気の観光地を紹介。",
    "url": "https://travel.example.jp/tokyo-guide"
  },
  {
    "title": "天気予報とは",
    "content": "天気予報は気象データに基づいて将来の天候を予測するものです。",
    "url": "https://wiki.example.jp/weather-forecast"
  },
  {
    "title": "雨具セール - 東京店舗一覧",
    "content": "折りたたみ傘が半額。東京各店舗で販売中。",
    "url": "https://shop.example.jp/umbrella"
  }
]
```

このリストがそのまま `composer.py` に渡り、プロンプトの「検索結果」ブロックになる。

### 3.3 Reranker の入力フォーマット（Issue #34 設計）

Reranker は **SearXNG 生 JSON ではなく**、上の `retriever` 出力 + **ユーザーの質問**を受け取る。

```python
query = "今日の東京の天気"

documents = [
  {"title": "...", "content": "...", "url": "..."},
  # ... 最大5件（SearXNG から来たもの）
]
```

Cross-Encoder 内部では、各候補を**「質問 + ドキュメント」ペア**としてスコアリングする（概念的には）:

```python
# モデル内部でこういうペアを評価
pairs = [
  ("今日の東京の天気", "東京の天気予報 - 今日・明日。東京都心部は晴れ..."),
  ("今日の東京の天気", "100年前の今日の東京の天気。1926年7月25日..."),
  ("今日の東京の天気", "東京観光ガイド - おすすめスポット。東京タワー..."),
  # ...
]
# → 各ペアに 0.0〜1.0 付近の関連度スコア
```

### 3.4 Reranker が返すフォーマット（Issue #34 設計）

Issue #34 では `RERANK_TOP_K=3` なので、**同じ dict 形式のまま、関連度順に並べ替えて上位 3 件**を返す想定。`composer.py` のインターフェースを変えずに済む形。

```python
[
  {
    "title": "東京の天気予報 - 今日・明日",
    "content": "東京都心部は晴れ、最高気温28度。午後から雲が広がる見込みです。",
    "url": "https://weather.example.jp/tokyo",
    "rerank_score": 0.94
  },
  {
    "title": "天気予報とは",
    "content": "天気予報は気象データに基づいて将来の天候を予測するものです。",
    "url": "https://wiki.example.jp/weather-forecast",
    "rerank_score": 0.31
  },
  {
    "title": "100年前の今日の東京の天気",
    "content": "1926年7月25日の東京は曇りでした。",
    "url": "https://news.example.jp/weather-history",
    "rerank_score": 0.18
  }
]
```

**落とされる候補**（関連度が低い）:

| title | なぜ落ちるか |
|-------|-------------|
| 東京観光ガイド | 「天気」より「観光」の話 |
| 雨具セール | 「東京」は含むが天気情報ではない |

`rerank_score` はデバッグ用の追加フィールド。`composer.py` は `title` / `content` だけ使えば動く。

### 3.5 最終的に LLM に渡る形（`composer.py`）

Reranker 後の 3 件が、プロンプトではこう展開される:

```
以下の検索結果を『絶対に事実』として扱い、ユーザーの質問に日本語のみで答えなさい。
...

検索結果：
- 東京の天気予報 - 今日・明日: 東京都心部は晴れ、最高気温28度。午後から雲が広がる見込みです。
- 天気予報とは: 天気予報は気象データに基づいて将来の天候を予測するものです。
- 100年前の今日の東京の天気: 1926年7月25日の東京は曇りでした。

質問：今日の東京の天気

回答：
```

Reranker なし（現状）だと 5 件全部入り、観光ガイドや傘セールも混ざる。

### 3.6 フォーマット比較（一覧）

| 段階 | 形式 | 件数 | 主なフィールド |
|------|------|------|----------------|
| **SearXNG 生 JSON** | HTTP レスポンス全体 | 多数（エンジン依存） | `query`, `results[]`, `suggestions` 等 |
| **SearXNG `results[i]`** | 1 件の検索ヒット | — | `title`, `url`, `content`, `engine`, `score`... |
| **`retriever.search_web()` 出力** | `list[dict]` | 最大 **5** | `title`, `content`, `url` |
| **Reranker 入力** | 上 + `query` 文字列 | 最大 5 | 同上 |
| **Reranker 出力（設計）** | `list[dict]` | 最大 **3** (`RERANK_TOP_K`) | `title`, `content`, `url`, (+ `rerank_score`) |
| **`composer` が使う部分** | プロンプト内テキスト | Reranker 後の件数 | `title` + `content` のみ |

### 3.7 `RERANKER_ENABLED=false` のとき

Issue #34 ではデフォルト無効なので、Reranker を挟まず **`retriever.search_web()` の 5 件がそのまま** `composer` に渡る。フォーマットは現状と同じ。

---

## 4. まとめ

| 項目 | 内容 |
|------|------|
| **目的** | 検索結果のノイズを減らし、LLM に渡す根拠の質を上げて回答精度（特に事実性）を改善する |
| **配置** | `retriever.py` 内の任意ステップ（SearXNG 取得後、composer 渡し前） |
| **SearXNG 出力** | 大きな JSON オブジェクト（`results[]` が本体） |
| **Retriever 出力** | `{title, content, url}` のリスト（最大 5 件） |
| **Reranker 出力（設計）** | 同形式のリストを関連度順に並べ替え、最大 3 件に絞る |
| **Composer への影響** | インターフェース変更なし（`title` / `content` をプロンプトに展開） |
