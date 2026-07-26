import os
import sys
from google import genai
from google.genai import types

# -------------------------------------------------------------------
# ツール1：ファイル読み込み
# -------------------------------------------------------------------
def read_file(filepath: str) -> str:
    """指定されたパスのファイル内容を読み込んでテキストとして返します。
    コードの修正前に既存の中身を確認するために必ず呼び出してください。
    """
    time.sleep(12)  # ← ★ここに挿入！(無料枠 5回/分 = 12秒に1回ペースにする)
    try:
        if not os.path.exists(filepath):
            return f"Error: File {filepath} does not exist."
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"📖 [Tool Executed] ファイルを読み込みました: {filepath}")
        return content
    except Exception as e:
        return f"Error reading {filepath}: {str(e)}"

# -------------------------------------------------------------------
# 2. ツール2：ファイル保存・上書き
# -------------------------------------------------------------------
def save_file(filepath: str, content: str) -> str:
    """指定されたパスにファイルの内容を書き込み・保存します。"""
    time.sleep(12)  # ← ★ここに挿入！(無料枠 5回/分 = 12秒に1回ペースにする)
    try:
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
            
        print(f"✅ [Tool Executed] ファイルを書き換えました: {filepath}")
        return f"Successfully saved {filepath}"
    except Exception as e:
        return f"Failed to save {filepath}: {str(e)}"

# -------------------------------------------------------------------
# 3. ツール3：ファイル・ディレクトリ一覧の取得
# -------------------------------------------------------------------
def list_files(directory: str = ".") -> str:
    """指定したディレクトリ内のファイルやフォルダの一覧を取得します。
    リポジトリの構造を把握したい時に利用してください。
    """
    time.sleep(12)  # ← ★ここに挿入！(無料枠 5回/分 = 12秒に1回ペースにする)
    try:
        file_list = []
        for root, dirs, files in os.walk(directory):
            # .git などの隠しフォルダや node_modules は除外
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules']
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                file_list.append(rel_path)
        return "\n".join(file_list[:100])  # 一度に多すぎないよう上限100件
    except Exception as e:
        return f"Error listing files: {str(e)}"

# -------------------------------------------------------------------
# 4. AGENTS.md の読み込み
# -------------------------------------------------------------------
def load_agents_md() -> str:
    agents_path = "AGENTS.md"
    if os.path.exists(agents_path):
        try:
            with open(agents_path, "r", encoding="utf-8") as f:
                return f"\n\n【プロジェクトの絶対制約ルール (AGENTS.md)】:\n{f.read()}\n"
        except Exception:
            pass
    return ""

# -------------------------------------------------------------------
# メイン処理
# -------------------------------------------------------------------
def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY がセットされていません。")
        sys.exit(1)

    prompt = os.getenv("TASK_PROMPT", "")
    agents_rules = load_agents_md()

    print(f"=== Antigravity Agent Task Started ===\n{prompt}\n")

    client = genai.Client()

    # 3つのツール（関数）をすべて Gemini に登録
    config = types.GenerateContentConfig(
        tools=[read_file, save_file, list_files],
        temperature=0.1,
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",  # または gemini-3.6-flash
            contents=(
                "あなたはGitHub Actions内で動く自律型コードエージェントです。\n"
                "【重要な行動手順】\n"
                "1. まず list_files や read_file ツールを使って対象ファイルの既存コードを確認してください。\n"
                "2. 既存のコードやコンテキストを完全に理解した上で、必要な部分のみを修正してください。\n"
                "3. 準備ができたら save_file ツールを使ってファイルを更新してください。\n"
                f"{agents_rules}"
                "\n【指示内容】:\n"
                f"{prompt}"
            ),
            config=config,
        )

        print("\n=== Agent Response ===")
        print(response.text)

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()