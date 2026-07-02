#!/usr/bin/env python3
"""
Composer module - RAG prompt composition
"""

from .audio_utils import log_init, log_ready


class Composer:
    def __init__(self):
        log_init("Composer")
        log_ready("Composer")

    def compose_prompt(self, query, search_results, history_summary=""):
        """Build prompt with search context (Enforce Japanese-only outputs to keep Piper stable).

        Args:
            query: The user's current question.
            search_results: List of dicts with 'title' and 'content' keys.
            history_summary: Optional condensed text of prior conversation turns.
        """
        history_block = ""
        if history_summary:
            history_block = f"これまでの会話履歴：\n{history_summary}"

        if search_results:
            context = "\n".join([
                f"- {r['title']}: {r['content']}"
                for r in search_results
            ])
            history_section = f"{history_block}\n\n" if history_block else ""
            prompt = f"""以下の検索結果を『絶対に事実』として扱い、ユーザーの質問に日本語のみで答えなさい。
回答にはアルファベット（英語の単語や文）を含めず、必要であればカタカナや日本語表現に翻訳して出力してください。

{history_section}検索結果：
{context}

質問：{query}

回答："""
        else:
            prefix = f"{history_block}\n\n" if history_block else ""
            prompt = f"{prefix}質問：{query}\n回答（日本語のみ、アルファベット禁止）："

        return prompt
