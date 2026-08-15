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
英語の単語や文、アルファベットの羅列は含めず、単位や略語はカタカナか日本語で書きなさい。
結論を先に述べ、文数は質問に合わせなさい。事実や数値は1文で足りれば1文で止め、多くても3文まで。挨拶やお礼は1〜2文。分からないことは「分かりません」と1〜2文。仕組みや手順の説明だけ3〜5文。
「それ」「さっきの」などの指示語は、これまでの会話を参照して解釈しなさい。
ユーザーが話題を変えたら、前の話題には触れずに新しい質問だけに答えなさい。
検索の有無や「検索結果」という言い方は回答に出してはいけません。
検索結果が質問と無関係な場合は、検索結果を使わず、これまでの会話とあなたの知識で答えなさい。
検索結果を使って答える場合は、検索結果に書かれていないことを推測で補ってはいけません。

{history_block}検索結果：
{context}

質問：{query}

回答："""
        else:
            prompt = f"""日本語のみで答えなさい。英語の単語や文、アルファベットの羅列は含めず、単位や略語はカタカナか日本語で書きなさい。
結論を先に述べ、文数は質問に合わせなさい。事実や数値は1文で足りれば1文で止め、多くても3文まで。挨拶やお礼は1〜2文。分からないことは「分かりません」と1〜2文。仕組みや手順の説明だけ3〜5文。
「それ」「さっきの」などの指示語は、これまでの会話を参照して解釈しなさい。
ユーザーが話題を変えたら、前の話題には触れずに新しい質問だけに答えなさい。
調べた内容の有無は回答に出してはいけません。

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
