#!/usr/bin/env python3
"""
Composer module - RAG prompt composition
"""

from .audio_utils import log_init, log_ready


class Composer:
    def __init__(self):
        log_init("Composer")
        log_ready("Composer")

    def compose_prompt(self, query, search_results, history_text: str = ""):
        """Build prompt with optional search context and conversation history.

        Args:
            query: The current user question.
            search_results: List of web search result dicts (may be empty).
            history_text: Pre-formatted conversation history string produced by
                ConversationHistory.format_for_prompt().  Empty string means no
                history is available.
        """
        parts = []

        if history_text:
            parts.append(
                "以下はこれまでの会話履歴です：\n"
                + history_text
                + "\n"
            )

        if search_results:
            context = "\n".join([
                f"- {r['title']}: {r['content']}"
                for r in search_results
            ])
            parts.append(
                "以下の検索結果を『絶対に事実』として扱い、ユーザーの質問に日本語のみで答えなさい。\n"
                "回答にはアルファベット（英語の単語や文）を含めず、必要であればカタカナや日本語表現に翻訳して出力してください。\n"
                f"\n検索結果：\n{context}\n"
            )
            parts.append(f"質問：{query}\n\n回答：")
        else:
            parts.append(f"質問：{query}\n回答（日本語のみ、アルファベット禁止）：")

        return "\n".join(parts)
