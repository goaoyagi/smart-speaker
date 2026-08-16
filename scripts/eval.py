#!/usr/bin/env python3
"""
評価ハーネス - 回答精度と応答時間を、音声を通さずテキスト入出力だけで測る。

マイク・スピーカー・GPIO には依存せず、本番と同じ ``run_text_turn``
（検索要否判定・クエリ書き換え → 検索 → 構成 → 生成 → 発話正規化）を
実行する。会話履歴も本番と同じ ConversationHistory で保持するため、
マルチターンの指示語解決や再復唱コマンドも評価できる。

評価は主観を混ぜず、機械的に判定できる項目だけを自動採点する。
- 日本語限定（発話正規化後にアルファベットが残っていないか）
- 期待キーワードの含有 / 禁止キーワードの非含有
- 文数（質問タイプに合わせた下限・上限）
- 再復唱コマンドが直前の回答をそのまま返しているか
- 応答時間（準備 / 検索 / 生成 / 合計）と回答文字数
- degrade（検索失敗・Reranking 失敗など）の warning 発生

回答本文はレポートに全文残すので、自動採点で拾えない品質は人が読んで判断する。

使い方（SearXNG と Ollama が起動している環境で実行する）:

    python3 scripts/eval.py                        # 全ケースを実行
    python3 scripts/eval.py --only fact-01,follow-01
    python3 scripts/eval.py --category followup
    python3 scripts/eval.py --repeat 3             # ばらつきを見る
    python3 scripts/eval.py --no-search            # 検索を通さず LLM 単体を見る
    python3 scripts/eval.py --baseline reports/eval_before.json
    python3 scripts/eval.py --list

任意の質問をその場で試すときは `--ask` を使う。評価ケースは読まず、
渡した質問をその順に1つの会話として実行する（複数指定で履歴の効き方を確かめられる）:

    python3 scripts/eval.py --ask "視力を良くする方法は？"
    python3 scripts/eval.py --ask "富士山の高さは？" --ask "その山はどこにある？"
"""

import argparse
import json
import logging
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# プロジェクトルートを sys.path に追加し、src/ を読めるようにする。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.brain import Brain  # noqa: E402
from src.composer import Composer  # noqa: E402
from src.conversation_history import ConversationHistory  # noqa: E402
from src.exceptions import GenerationError  # noqa: E402
from src.query_prep import QueryPrep  # noqa: E402
from src.retriever import Retriever  # noqa: E402
from src.text_turn import run_text_turn  # noqa: E402

DEFAULT_CASES_PATH = PROJECT_ROOT / "scripts" / "eval_cases.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

SCHEMA_VERSION = 1

# 全角も含めてラテン文字を拾う（TTS が読めない英字混入の検出）。
_LATIN_RUN = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]+")
_SENTENCE_END = re.compile(r"[。！？!?]+")

# レポートに残す設定値。実行条件が違うレポートを取り違えないための記録。
_CONFIG_KEYS = (
    "OLLAMA_API_URL",
    "OLLAMA_MODEL",
    "OLLAMA_NUM_CTX",
    "OLLAMA_NUM_PREDICT",
    "OLLAMA_TEMPERATURE",
    "OLLAMA_REPEAT_PENALTY",
    "OLLAMA_KEEP_ALIVE",
    "OLLAMA_SYSTEM_PROMPT",
    "OLLAMA_AUX_NUM_PREDICT",
    "OLLAMA_AUX_TEMPERATURE",
    "QUERY_PREP_ENABLED",
    "CONTEXT_CHAR_BUDGET",
    "CHAR_TO_TOKEN_RATIO",
    "SEARXNG_URL",
    "SEARCH_CANDIDATE_LIMIT",
    "RERANKER_ENABLED",
    "RERANKER_MODEL_ID",
    "RERANK_TOP_K",
    "RERANK_MIN_SCORE",
    "FETCH_PAGE_ENABLED",
    "FETCH_PAGE_TOP_N",
    "FETCH_PAGE_TIMEOUT",
    "PASSAGE_CHARS",
    "CONVERSATION_MAX_TURNS",
    "CONVERSATION_ANSWER_CLIP",
)


# --- 採点（純粋関数。外部サービスに触らない） ---------------------------------


def count_sentences(text):
    """句点・感嘆符・疑問符で区切った文の数を返す。"""
    if not text:
        return 0
    return len([part for part in _SENTENCE_END.split(text) if part.strip()])


def find_latin_runs(text):
    """回答に含まれるラテン文字の連なりを返す（空なら日本語限定を満たす）。"""
    return _LATIN_RUN.findall(text or "")


def grade_answer(turn_spec, answer, previous_answer=None):
    """1ターンの回答を採点し、``(passed, failures)`` を返す。

    ``failures`` は人が読める失敗理由の一覧。空なら合格。
    """
    failures = []

    latin = find_latin_runs(answer)
    if latin:
        failures.append(f"アルファベット混入: {', '.join(latin[:5])}")

    sentences = count_sentences(answer)
    min_sentences = turn_spec.get("min_sentences", 2)
    max_sentences = turn_spec.get("max_sentences")
    if min_sentences and sentences < min_sentences:
        failures.append(f"文数が不足: {sentences}文 < {min_sentences}文")
    if max_sentences and sentences > max_sentences:
        failures.append(f"文数が過多: {sentences}文 > {max_sentences}文")

    missing = [
        keyword
        for keyword in turn_spec.get("expect_keywords", [])
        if keyword not in answer
    ]
    if missing:
        failures.append(f"期待キーワード欠落: {', '.join(missing)}")

    for group in turn_spec.get("expect_any", []):
        if not any(keyword in answer for keyword in group):
            failures.append(f"期待キーワード（いずれか）欠落: {' / '.join(group)}")

    hit_denied = [
        keyword
        for keyword in turn_spec.get("deny_keywords", [])
        if keyword in answer
    ]
    if hit_denied:
        failures.append(f"禁止キーワード出現: {', '.join(hit_denied)}")

    if turn_spec.get("expect_repeat"):
        if previous_answer is None:
            failures.append("再復唱の比較対象となる直前の回答がない")
        elif answer.strip() != previous_answer.strip():
            failures.append("再復唱コマンドが直前の回答と一致しない")

    return (not failures), failures


def summarize(turn_results):
    """全ターンの結果から集計値を作る。"""
    total = len(turn_results)
    completed = [turn for turn in turn_results if turn.get("answer") is not None]
    passed = [turn for turn in turn_results if turn.get("passed")]

    def _stats(values):
        if not values:
            return None
        return {
            "mean": round(statistics.fmean(values), 1),
            "median": round(statistics.median(values), 1),
            "max": round(max(values), 1),
        }

    by_category = {}
    for turn in turn_results:
        bucket = by_category.setdefault(
            turn["category"], {"turns": 0, "passed": 0}
        )
        bucket["turns"] += 1
        if turn.get("passed"):
            bucket["passed"] += 1
    for bucket in by_category.values():
        bucket["pass_rate"] = round(bucket["passed"] / bucket["turns"], 3)

    return {
        "turns": total,
        "errors": total - len(completed),
        "passed": len(passed),
        "pass_rate": round(len(passed) / total, 3) if total else 0.0,
        "degraded": sum(1 for turn in turn_results if turn.get("degraded")),
        "latin_violations": sum(1 for turn in turn_results if turn.get("latin_runs")),
        "answer_chars": _stats([turn["answer_chars"] for turn in completed]),
        "sentences": _stats([turn["sentences"] for turn in completed]),
        "search_ms": _stats([turn["search_ms"] for turn in completed]),
        "prep_ms": _stats([turn.get("prep_ms", 0.0) for turn in completed]),
        "generate_ms": _stats([turn["generate_ms"] for turn in completed]),
        "total_ms": _stats([turn["total_ms"] for turn in completed]),
        "warnings": sum(len(turn.get("warnings", [])) for turn in turn_results),
        "by_category": by_category,
    }


def diff_summaries(baseline, current):
    """ベースラインとの差分を、比較しやすい平坦な辞書で返す。"""
    diff = {}
    for key in ("pass_rate", "latin_violations", "degraded", "warnings"):
        before = baseline.get(key)
        after = current.get(key)
        if before is None or after is None:
            continue
        diff[key] = {"before": before, "after": after, "delta": round(after - before, 3)}
    for key in ("answer_chars", "sentences", "prep_ms", "generate_ms", "total_ms"):
        before = (baseline.get(key) or {}).get("median")
        after = (current.get(key) or {}).get("median")
        if before is None or after is None:
            continue
        diff[f"{key}_median"] = {
            "before": before,
            "after": after,
            "delta": round(after - before, 1),
        }
    return diff


# --- 実行 ---------------------------------------------------------------------


class WarningCollector(logging.Handler):
    """`src` 配下の WARNING 以上を集め、degrade の発生を可視化する。"""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        if record.name.startswith("src."):
            self.records.append(record.getMessage())

    def drain(self):
        collected = self.records
        self.records = []
        return collected


def build_adhoc_case(questions):
    """`--ask` で渡された質問を、1ケース（連続する会話）に組み立てる。

    回答を目で確かめるためのモードなので、文数やキーワードの期待値は置かない。
    アルファベット混入だけは常に検出する（TTS が読めないため）。
    """
    return {
        "id": "ask",
        "category": "ad_hoc",
        "note": "--ask で渡された質問",
        "turns": [
            {"input": question, "min_sentences": 0} for question in questions
        ],
    }


def load_cases(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    cases = data.get("cases", [])
    if not cases:
        raise ValueError(f"評価ケースが空です: {path}")
    return cases


def select_cases(cases, only=None, categories=None):
    selected = cases
    if only:
        wanted = {case_id.strip() for case_id in only.split(",") if case_id.strip()}
        selected = [case for case in selected if case["id"] in wanted]
        unknown = wanted - {case["id"] for case in cases}
        if unknown:
            raise ValueError(f"存在しないケースID: {', '.join(sorted(unknown))}")
    if categories:
        wanted = {name.strip() for name in categories.split(",") if name.strip()}
        selected = [case for case in selected if case.get("category") in wanted]
    if not selected:
        raise ValueError("条件に一致する評価ケースがありません")
    return selected


def run_turn(components, history, case, turn_spec, run_index, use_search, warnings):
    """1ターンを実行し、計測値と採点結果を返す。"""
    retriever, composer, brain, query_prep = components
    query = turn_spec["input"]
    previous_answer = history.last_answer()

    result = {
        "case_id": case["id"],
        "category": case.get("category", "uncategorized"),
        "run": run_index,
        "input": query,
        "answer": None,
        "answer_chars": 0,
        "sentences": 0,
        "search_hits": 0,
        "search_query": query,
        "skipped_search": False,
        "prep_ms": 0.0,
        "search_ms": 0.0,
        "generate_ms": 0.0,
        "total_ms": 0.0,
        "latin_runs": [],
        "passed": False,
        "degraded": False,
        "failures": [],
        "warnings": [],
        "path": "generate",
    }

    started = time.perf_counter()

    # 再復唱コマンドは本番と同じく検索も生成も通さない経路を測る。
    if history.is_repeat_command(query):
        answer = history.last_answer() or "まだお答えできる内容がありません。"
        result["path"] = "repeat"
    else:
        try:
            turn = run_text_turn(
                query,
                history.as_messages(),
                retriever=retriever,
                composer=composer,
                brain=brain,
                query_prep=query_prep,
                use_search=use_search and not turn_spec.get("skip_search"),
            )
        except GenerationError as error:
            result["total_ms"] = (time.perf_counter() - started) * 1000
            result["failures"].append(f"生成失敗: {error}")
            result["warnings"] = warnings.drain()
            return result

        answer = turn["answer"]
        result["search_hits"] = len(turn["search_results"])
        result["search_query"] = turn["search_query"]
        result["skipped_search"] = turn["skipped_search"]
        result["prep_ms"] = turn["prep_ms"]
        result["search_ms"] = turn["search_ms"]
        result["generate_ms"] = turn["generate_ms"]
        if turn["degraded"]:
            # 本番と同じく非致命。ただし検索なしの回答は他の実行と比較できないため、
            # degrade として記録し、そのターンは不合格として扱う。
            result["degraded"] = True
            result["failures"].append(
                f"検索失敗のため計測が不完全: {turn['degrade_reason']}"
            )

    result["total_ms"] = (time.perf_counter() - started) * 1000
    result["answer"] = answer
    result["answer_chars"] = len(answer)
    result["sentences"] = count_sentences(answer)
    result["latin_runs"] = find_latin_runs(answer)
    result["passed"], failures = grade_answer(turn_spec, answer, previous_answer)
    result["failures"].extend(failures)
    if result["failures"]:
        result["passed"] = False
    result["warnings"] = warnings.drain()

    # 本番と同じく、発話まで済んだターンだけを履歴に積む。
    if result["path"] == "generate":
        history.add(query, answer)

    return result


def run_case(components, case, run_index, use_search, warnings, verbose=True, adhoc=False):
    """1ケース（複数ターン）を、新しい会話履歴で実行する。"""
    history = ConversationHistory()
    turn_results = []
    for turn_spec in case["turns"]:
        result = run_turn(
            components, history, case, turn_spec, run_index, use_search, warnings
        )
        turn_results.append(result)
        if verbose:
            print_turn(result, adhoc=adhoc)
    return turn_results


def print_turn(result, adhoc=False):
    if adhoc:
        print(f"[ask] run{result['run']} ({result['path']})")
    else:
        mark = "PASS" if result["passed"] else "FAIL"
        print(f"[{mark}] {result['case_id']} run{result['run']} ({result['path']})")
    print(f"  Q: {result['input']}")
    answer = result["answer"]
    print(f"  A: {answer if answer is not None else '(回答なし)'}")
    print(
        "  "
        f"{result['answer_chars']}文字 / {result['sentences']}文 / "
        f"検索{result['search_hits']}件"
        f"{'（スキップ）' if result.get('skipped_search') else ''} / "
        f"準備{result.get('prep_ms', 0.0):.0f}ms + "
        f"検索{result['search_ms']:.0f}ms + 生成{result['generate_ms']:.0f}ms "
        f"= 合計{result['total_ms']:.0f}ms"
    )
    for failure in result["failures"]:
        print(f"  - {failure}")
    for warning in result["warnings"]:
        print(f"  ! warning: {warning}")
    print()


def print_summary(summary, adhoc=False):
    print("=" * 60)
    print("集計")
    print("=" * 60)
    # --ask には期待値がないので、合格率とカテゴリ別は出さない。
    if not adhoc:
        print(
            f"合格 {summary['passed']}/{summary['turns']} ターン "
            f"(pass_rate {summary['pass_rate']:.1%})"
        )
    if summary["errors"]:
        print(f"回答が得られなかったターン: {summary['errors']}")
    if summary["degraded"]:
        print(
            f"検索失敗で degrade したターン: {summary['degraded']}"
            "（SearXNG の状態を確認してから再測定すること）"
        )
    print(f"アルファベット混入: {summary['latin_violations']} ターン")
    print(f"degrade warning: {summary['warnings']} 件")
    for key, label in (
        ("answer_chars", "回答文字数"),
        ("sentences", "文数"),
        ("prep_ms", "準備(ms)"),
        ("search_ms", "検索(ms)"),
        ("generate_ms", "生成(ms)"),
        ("total_ms", "合計(ms)"),
    ):
        stats = summary.get(key)
        if stats:
            print(
                f"{label}: 中央値 {stats['median']} / 平均 {stats['mean']} / 最大 {stats['max']}"
            )
    if adhoc:
        return
    print()
    print("カテゴリ別")
    for category, bucket in sorted(summary["by_category"].items()):
        print(
            f"  {category}: {bucket['passed']}/{bucket['turns']} "
            f"({bucket['pass_rate']:.1%})"
        )


def print_diff(diff):
    print()
    print("=" * 60)
    print("ベースラインとの差分")
    print("=" * 60)
    for key, values in diff.items():
        sign = "+" if values["delta"] > 0 else ""
        print(f"  {key}: {values['before']} → {values['after']} ({sign}{values['delta']})")


def config_snapshot():
    return {key: getattr(config, key) for key in _CONFIG_KEYS}


def render_markdown(report):
    summary = report["summary"]
    lines = [
        "# 評価レポート",
        "",
        f"- 実行: {report['started_at']}",
        f"- モデル: `{report['config']['OLLAMA_MODEL']}`",
        f"- 検索: {'有効' if report['use_search'] else '無効'}",
        f"- 合格: {summary['passed']}/{summary['turns']} ({summary['pass_rate']:.1%})",
        "",
        "## 集計",
        "",
        "| 指標 | 中央値 | 平均 | 最大 |",
        "| --- | --- | --- | --- |",
    ]
    for key, label in (
        ("answer_chars", "回答文字数"),
        ("sentences", "文数"),
        ("prep_ms", "準備(ms)"),
        ("search_ms", "検索(ms)"),
        ("generate_ms", "生成(ms)"),
        ("total_ms", "合計(ms)"),
    ):
        stats = summary.get(key)
        if stats:
            lines.append(
                f"| {label} | {stats['median']} | {stats['mean']} | {stats['max']} |"
            )
    lines += ["", "## ターン別", "", "| ケース | 判定 | 質問 | 文字数 | 文数 | 合計ms | 失敗理由 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for turn in report["turns"]:
        lines.append(
            f"| {turn['case_id']} | {'PASS' if turn['passed'] else 'FAIL'} "
            f"| {turn['input']} | {turn['answer_chars']} | {turn['sentences']} "
            f"| {turn['total_ms']:.0f} | {'; '.join(turn['failures']) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="回答精度と応答時間をテキスト入出力だけで評価する"
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="評価ケースのJSON")
    parser.add_argument(
        "--ask",
        action="append",
        metavar="質問",
        help="評価ケースを使わず、渡した質問の回答を確かめる（複数指定すると履歴を繋いだ連続ターンになる）",
    )
    parser.add_argument("--only", help="実行するケースIDをカンマ区切りで指定")
    parser.add_argument("--category", help="実行するカテゴリをカンマ区切りで指定")
    parser.add_argument("--repeat", type=int, default=1, help="各ケースの実行回数")
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="SearXNG 検索を通さず、LLM 単体の応答を評価する",
    )
    parser.add_argument("--out", help="レポートJSONの出力先（既定: reports/eval_<日時>.json）")
    parser.add_argument("--markdown", help="Markdown サマリの出力先")
    parser.add_argument("--baseline", help="比較するレポートJSON")
    parser.add_argument("--quiet", action="store_true", help="ターンごとの出力を省略する")
    parser.add_argument("--list", action="store_true", help="評価ケースの一覧を表示して終了")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    adhoc = bool(args.ask)
    if adhoc:
        selected = [build_adhoc_case(args.ask)]
    else:
        cases = load_cases(args.cases)

    if args.list and adhoc:
        raise SystemExit("--ask と --list は同時に使えません")

    if args.list:
        for case in cases:
            print(
                f"{case['id']:<12} {case.get('category', ''):<14} "
                f"{len(case['turns'])}ターン  {case.get('note', '')}"
            )
        return 0

    if not adhoc:
        selected = select_cases(cases, args.only, args.category)

    warnings = WarningCollector()
    logging.getLogger().addHandler(warnings)

    brain = Brain()
    components = (Retriever(), Composer(), brain, QueryPrep(brain))
    use_search = not args.no_search

    started_at = datetime.now(timezone.utc)
    turn_results = []
    for run_index in range(1, args.repeat + 1):
        for case in selected:
            turn_results.extend(
                run_case(
                    components,
                    case,
                    run_index,
                    use_search,
                    warnings,
                    verbose=not args.quiet,
                    adhoc=adhoc,
                )
            )
    finished_at = datetime.now(timezone.utc)

    summary = summarize(turn_results)
    report = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_s": round((finished_at - started_at).total_seconds(), 1),
        "cases_path": None if adhoc else str(args.cases),
        "adhoc": adhoc,
        "use_search": use_search,
        "repeat": args.repeat,
        "config": config_snapshot(),
        "turns": turn_results,
        "summary": summary,
    }

    print_summary(summary, adhoc=adhoc)

    if args.baseline:
        with open(args.baseline, encoding="utf-8") as handle:
            baseline = json.load(handle)
        print_diff(diff_summaries(baseline.get("summary", {}), summary))

    # --ask は手元で回答を見るためのモードなので、明示されない限りレポートを残さない。
    if args.out or not adhoc:
        out_path = Path(args.out) if args.out else (
            DEFAULT_REPORT_DIR / f"eval_{started_at.strftime('%Y%m%d_%H%M%S')}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print()
        print(f"レポートを保存しました: {out_path}")

    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Markdown サマリを保存しました: {markdown_path}")

    # --ask には期待値がないので、回答が得られたかだけを終了コードにする。
    if adhoc:
        return 0 if summary["errors"] == 0 else 1

    # 1ターンでも落ちたら非ゼロ終了（改善前後の比較を自動化しやすくする）。
    return 0 if summary["passed"] == summary["turns"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
