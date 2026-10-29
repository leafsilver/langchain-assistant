from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.model import create_model


def main():
    model = create_model()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名耐心的编程老师，回答要准确、通俗，并给出一个简单例子。",
            ),
            (
                "human",
                "请向{learner}解释{topic}，控制在300字以内。",
            ),
        ]
    )

    output_parser = StrOutputParser()
    chain = prompt | model | output_parser

    topic = input("请输入想了解的编程概念：").strip()

    if not topic:
        print("主题不能为空。")
        return

    result = chain.invoke(
        {
            "learner": "学过 Python 基础语法的初学者",
            "topic": topic,
        }
    )

    print("\n模型回答：")
    print(result)


if __name__ == "__main__":
    main()