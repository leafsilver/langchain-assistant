from app.model import create_model


model = create_model()

response = model.invoke("请用一句话解释什么是 LangChain。")

print(response.content)
