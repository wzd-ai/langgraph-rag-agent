"""
项目入口：接收用户输入，调用 Agent，打印最终回复。
"""
from dotenv import load_dotenv
load_dotenv()

from src.graph import app
from src.state import OverallState


def main():
    print("=" * 50)
    print("知识库问答 Agent 已启动（输入 q 退出）")
    print("=" * 50)

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in ("q", "quit", "exit"):
            print("再见！")
            break
        if not user_input:
            continue

        # 初始化 State
        initial_state: OverallState = {
            "user_query": user_input,
            "original_query": user_input,
            "intents": [],
            "need_escalate": False,
            "results": [],
            "final_answer": "",
            "sub_queries": [],   #  新增
            "retry_count": 0,    #  新增
        }

        try:
            result = app.invoke(initial_state)
            print(f"\nAgent：{result['final_answer']}")
        except Exception as e:
            print(f"\n[错误] {e}")


if __name__ == "__main__":
    main()