from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from app.model import create_model
from app.tools import calculate_total


def main():
    model = create_model()
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[calculate_total],
        system_prompt=(
            "你是一名耐心、准确的智能助手。"
            "遇到商品总价问题时使用计算工具，不要自己猜测计算结果。"
        ),
        checkpointer=checkpointer,
    )

    thread_config = {
        "configurable": {
            "thread_id": "terminal-session",
        }
    }

    print("智能助手已启动，输入 exit 结束对话。")

    while True:
        question = input("\n你：").strip()

        if question.lower() == "exit":
            print("对话已结束。")
            break

        if not question:
            print("问题不能为空。")
            continue

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            },
            config=thread_config,
        )

        final_message = result["messages"][-1]

        print("\n助手：")
        print(final_message.content)


if __name__ == "__main__":
    main()