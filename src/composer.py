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
