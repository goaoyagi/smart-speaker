# issue 草案

`docs/現状分析と改善案_実装順序比較.html`「4案を統合した場合の実装順序」の順序1〜3を、そのまま起票できる形にした草案。内容を確定させたうえで GitHub issue として起票する。

| 順序 | 草案 | 概要 | 依存追加 |
| --- | --- | --- | --- |
| 1 | [01-ollama-generation-options.md](01-ollama-generation-options.md) | `brain.py` に `options`（`num_ctx` 等）と `system` を追加し、`config.py` ＋ `.env.example` 経由で設定化 | なし |
| 2 | [02-composer-prompt-strengthening.md](02-composer-prompt-strengthening.md) | `composer.py` のプロンプト強化（文量・構成・指示語解釈・無関係な検索結果の扱い） | なし |
| 3 | [03-api-chat-migration-and-token-budget.md](03-api-chat-migration-and-token-budget.md) | `/api/chat` 移行と履歴の messages 化、履歴 turns/clip の拡張、`prompt[:10000]` のトークン予算管理への置換 | なし |

順序1と2は独立して着手できるが、**効果を切り分けるために別 PR に分ける**（AGENTS.md「1 PR ＝ 1目的」）。順序3は1・2の効果測定後に着手する。

各草案の「決めるべき数値」「決めるべき設計判断」の節は、着手前に確定させる必要がある。
