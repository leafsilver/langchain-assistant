from langchain.tools import tool


@tool
def calculate_total(price: float, quantity: int) -> float:
    """根据商品单价和购买数量计算总价。"""
    return round(price * quantity, 2)