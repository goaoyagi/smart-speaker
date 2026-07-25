import os
import sys
from google import genai
from google.genai import types

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ エラー: GEMINI_API_KEY がセットされていません。GitHub Secretsを確認してください。")
        sys.exit(1)

    prompt = os.getenv("TASK_PROMPT", "コードベースの不具合を修正してください。")
    print("=== Antigravity Agent Task Started ===")
    print(f"Instruction:\n{prompt}\n")

    try:
        # GEMINI_API_KEY 環境変数を自動で読み込んでクライアントを作成
        client = genai.Client()

        # トークン制限・温度パラメーター・コード実行ツールの設定
        config = types.GenerateContentConfig(
            max_output_tokens=2048,
            temperature=0.2,
            tools=[{"code_execution": {}}]  # Pythonコード実行ツールを有効化
        )

        # 【修正】 client.models.generate_content メソッドを使用
        response = client.models.generate_content(
            model="gemini-3.6-flash",  # または "gemini-2.0-flash" / "gemini-1.5-pro"
            contents=(
                "あなたはGitHub Actions内で動く自律型開発エージェントです。\n"
                "【制約ルール】\n"
                "1. 変更は指示された最小限の範囲・対象ファイルのみにとどめてください。\n"
                "2. 挨拶や長文の解説は不要です。実際に修正した内容と結果のまとめのみ簡潔に出力してください。\n"
                "3. 無駄な試行錯誤を避けるため、効率よく最短ステップで修正を完了させてください。\n\n"
                f"【指示内容】:\n{prompt}"
            ),
            config=config,
        )

        print("\n=== Agent Task Finished ===")
        print(response.text)

    except Exception as e:
        print(f"❌ エージェント実行中にエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()