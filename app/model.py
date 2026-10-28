import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def create_model():
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    model_name = os.getenv("DEEPSEEK_MODEL")

    if not api_key:
        raise RuntimeError("缺少环境变量：DEEPSEEK_API_KEY")

    if not base_url:
        raise RuntimeError("缺少环境变量：DEEPSEEK_BASE_URL")

    if not model_name:
        raise RuntimeError("缺少环境变量：DEEPSEEK_MODEL")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        extra_body={
            "thinking": {
                "type": "disabled",
            },
        },
    )
