from langchain.agents import create_agent

from app.model import create_model
from app.tools import calculate_total


def main():
    model = create_model()

    agent = create_agent(
        model=model,
        tools=[calculate_total],
        system_prompt=(
            "你是一名耐心、准确的智能助手。"
            "遇到商品总价问题时使用计算工具，不要自己猜测计算结果。"
        ),
    )

    question = input("请输入问题：").strip()

    if not question:
        print("问题不能为空。")
        return

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    print("\nAgent 运行过程：")

    for message in result["messages"]:
        print(f"\n消息类型：{message.type}")
        print(f"消息内容：{message.content}")

        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            print(f"工具调用：{tool_calls}")

    final_message = result["messages"][-1]

    print("\n助手回答：")
    print(final_message.content)


if __name__ == "__main__":
    main()