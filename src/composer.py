#!/usr/bin/env python3
"""
Composer module - RAG prompt composition
"""

import logging

from .audio_utils import log_init, log_ready
from .config import CONTEXT_CHAR_BUDGET, OLLAMA_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def format_search_context(search_results):
    """Render search hits as a bullet list for the prompt / user message."""
    return "\n".join(
        f"- {result['title']}: {result['content']}"
        for result in search_results
    )


def clip_search_results(search_results, char_budget=None):
    """Drop search hits from the back until the rendered context fits the budget."""
    if char_budget is None:
        char_budget = CONTEXT_CHAR_BUDGET
    kept = list(search_results)
    while kept and len(format_search_context(kept)) > char_budget:
        kept.pop()
    if len(kept) < len(search_results):
        logger.warning(
            "Dropped %s search results to fit CONTEXT_CHAR_BUDGET (%s)",
            len(search_results) - len(kept),
            char_budget,
        )
    return kept


class Composer:
    def __init__(self):
        log_init("Composer")
        log_ready("Composer")

    def compose_prompt(self, query, search_results, history_context=""):
        """Build prompt with search context (Enforce Japanese-only outputs to keep Piper stable)

        ``history_context`` is an optional condensed summary of prior turns; when
        empty the prompt is identical to the single-turn form.

        Kept for rollback from ``main.py``; the live path uses ``compose_messages``.
        """
        history_block = ""
        if history_context:
            history_block = f"これまでの会話：\n{history_context}\n\n"

        if search_results:
            context = format_search_context(search_results)
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
        else:
            prompt = f"""日本語のみで、アルファベット（英語の単語や文）を含めずに答えなさい。
回答は3〜5文で、結論を先に述べてから補足を加える形で、具体的に説明しなさい。
「それ」「さっきの」などの指示語は、これまでの会話を参照して解釈しなさい。

{history_block}質問：{query}

回答："""

        return prompt

    def compose_messages(self, query, search_results, history_messages=None):
        """Build /api/chat messages with system, history, and the current question."""
        clipped = clip_search_results(search_results)
        messages = [{"role": "system", "content": OLLAMA_SYSTEM_PROMPT}]
        if history_messages:
            messages.extend(history_messages)

        if clipped:
            user_content = f"検索結果：\n{format_search_context(clipped)}\n\n質問：{query}"
        else:
            user_content = f"質問：{query}"

        messages.append({"role": "user", "content": user_content})
        return messages
