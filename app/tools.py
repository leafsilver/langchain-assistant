from langchain.tools import tool

from app.knowledge import get_vector_store


@tool
def calculate_total(price: float, quantity: int) -> float:
    """根据商品单价和购买数量计算总价。"""
    return round(price * quantity, 2)


@tool
def search_course_handbook(question: str) -> str:
    """查询训练营手册。凡是涉及训练营的任何事实或规定，
    包括基本信息、代号、时间、地点、作业、设备借用和结业要求，都应使用此工具。"""
    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        question,
        k=2,
    )

    results = []

    for document in documents:
        results.append(
            f"来源：{document.metadata['source']}\n"
            f"内容：{document.page_content}"
        )

    return "\n\n".join(results)