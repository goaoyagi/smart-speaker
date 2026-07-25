import os
import sys
from google import genai
from google.genai import types

def main():
    # GitHub Actionsの環境変数から指示内容（プロンプト）を取得
    prompt = os.getenv("TASK_PROMPT", "コードベースの不具合を修正し、コード品質を向上させてください。")
    
    print(f"=== Antigravity Agent Task Started ===")
    print(f"Instruction: {prompt}\n")

    # GEMINI_API_KEY を自動読み込みしてクライアント作成
    client = genai.Client()

    # 【対策①-1】 1回の応答トークン数上限と生成パラメータの抑制
    config = types.GenerateContentConfig(
        max_output_tokens=2048,  # 1回の応答で出力する最大トークン数を制限（約1,500文字程度）
        temperature=0.2,         # 応答のランダム性を下げ、無駄な試行錯誤・迷いを防止
    )

    # 【対策①-2】 プロンプトによる制約 ＆ ループ上限の指定
    interaction = client.create(
        agent="antigravity-preview-05-2026",
        input=(
            "あなたはGitHub Actions内で動く自律型開発エージェントです。\n"
            "【制約ルール】\n"
            "1. 変更は指示された最小限の範囲・対象ファイルのみにとどめてください。\n"
            "2. 挨拶や長文の解説は不要です。実際に修正した内容と結果のまとめのみ簡潔に出力してください。\n"
            "3. 無駄な試行錯誤を避けるため、効率よく最短ステップで修正を完了させてください。\n\n"
            f"【指示内容】:\n{prompt}"
        ),
        config={
            "tools": [{"code_execution": {}}],
            "generation_config": config,
            "max_steps": 5,  # エージェントの思考・コマンド実行ループを最大5回までに強制制限
        }
    )

    print("\n=== Agent Task Finished ===")
    print(interaction.output_text)

if __name__ == "__main__":
    main()