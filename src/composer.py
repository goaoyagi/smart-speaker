#!/usr/bin/env python3
"""
Composer module - RAG prompt composition
"""

from .audio_utils import log_init, log_ready


class Composer:
    def __init__(self):
        log_init("Composer")
        log_ready("Composer")

    def compose_prompt(self, query, search_results, history=None, history_text=None):
        """Build prompt with search context and optional conversation history.

        Args:
            query: The user's current question.
            search_results: List of dicts with 'title' and 'content' keys.
            history: Optional :class:`ConversationHistory` instance.  When
                provided and non-empty, past turns are prepended to the prompt
                so the model can maintain conversational context.
            history_text: Optional pre-computed history string (e.g. an LLM
                summary produced by :class:`~src.history_summarizer.HistorySummarizer`).
                When supplied, this string is used *instead of* calling
                ``history.format_for_prompt()``, allowing Phase B summarization
                to be injected without altering the prompt structure.

        Returns:
            A formatted prompt string for the LLM.
        """
        history_block = ""
        if history_text is not None and history_text.strip():
            history_block = "これまでの会話履歴（要約）：\n" + history_text + "\n\n"
        elif history is not None and not history.is_empty():
            history_block = (
                "これまでの会話履歴：\n"
                + history.format_for_prompt()
                + "\n\n"
            )

        if search_results:
            context = "\n".join([
                f"- {r['title']}: {r['content']}"
                for r in search_results
            ])
            prompt = (
                f"{history_block}"
                f"以下の検索結果を『絶対に事実』として扱い、ユーザーの質問に日本語のみで答えなさい。\n"
                f"回答にはアルファベット（英語の単語や文）を含めず、必要であればカタカナや日本語表現に翻訳して出力してください。\n"
                f"\n"
                f"検索結果：\n"
                f"{context}\n"
                f"\n"
                f"質問：{query}\n"
                f"\n"
                f"回答："
            )
        else:
            prompt = (
                f"{history_block}"
                f"質問：{query}\n"
                f"回答（日本語のみ、アルファベット禁止）："
            )

        return prompt
