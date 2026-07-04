#!/usr/bin/env python3
"""
Composer module - RAG prompt composition
"""

from .audio_utils import log_init, log_ready


class Composer:
    def __init__(self):
        log_init("Composer")
        log_ready("Composer")

    def compose_prompt(self, query, search_results, history_context: str = ""):
        """Build prompt with search context and optional conversation history.

        Args:
            query: The user's current question.
            search_results: List of dicts with 'title' and 'content' keys.
            history_context: Pre-formatted or LLM-summarized history string.
                When non-empty it is prepended to the prompt so the model can
                maintain conversational context.  Pass an empty string (the
                default) to omit history and reproduce the original behaviour.

        Returns:
            A formatted prompt string for the LLM.
        """
        history_block = ""
        if history_context:
            history_block = f"これまでの会話の要約：\n{history_context}\n\n"

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
