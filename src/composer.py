#!/usr/bin/env python3
"""
Composer module - RAG prompt composition
"""

from .audio_utils import log_init, log_ready


class Composer:
    def __init__(self):
        log_init("Composer")
        log_ready("Composer")

    def compose_prompt(self, query, search_results, history_context=""):
        """Build prompt with search context (Enforce Japanese-only outputs to keep Piper stable)

        ``history_context`` is an optional condensed summary of prior turns; when
        empty the prompt is identical to the single-turn form.
        """
        history_block = ""
        if history_context:
            history_block = f"これまでの会話：\n{history_context}\n\n"

        if search_results:
            context = "\n".join([
                f"- {r['title']}: {r['content']}"
                for r in search_results
            ])
            prompt = f"""以下の検索結果を『絶対に事実』として扱い、ユーザーの質問に日本語のみで答えなさい。
回答にはアルファベット（英語の単語や文）を含めず、必要であればカタカナや日本語表現に翻訳して出力してください。

{history_block}検索結果：
{context}

質問：{query}

回答："""
        else:
            prompt = f"{history_block}質問：{query}\n回答（日本語のみ、アルファベット禁止）："

        return prompt
