from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_FILE = PROJECT_ROOT / "data" / "course_handbook.md"
CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_db"


def load_and_split_documents():
    """读取知识库文件，并将内容分割成多个 Document。"""
    text = KNOWLEDGE_FILE.read_text(encoding="utf-8")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    documents = text_splitter.create_documents(
        [text],
        metadatas=[{"source": KNOWLEDGE_FILE.name}],
    )

    return documents


def get_vector_store():
    """打开或创建 Chroma 数据库操作对象。"""
    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:0.6b",
        base_url="http://127.0.0.1:11434",
    )

    return Chroma(
        collection_name="course_handbook",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIRECTORY),
    )


def build_vector_store(documents):
    """将分块后的文档生成向量，并写入 Chroma。"""
    vector_store = get_vector_store()

    document_ids = [
        f"course-handbook-{index}"
        for index in range(1, len(documents) + 1)
    ]

    vector_store.add_documents(
        documents=documents,
        ids=document_ids,
    )

    return vector_store


if __name__ == "__main__":
    documents = load_and_split_documents()
    vector_store = build_vector_store(documents)

    question = "作业最晚什么时候提交？"

    results = vector_store.similarity_search(
        question,
        k=2,
    )

    print(f"共生成 {len(documents)} 个文本块。")
    print("问题：", question)
    print(f"检索到 {len(results)} 个相关文本块：")

    for index, document in enumerate(results, start=1):
        print(f"\n--- 检索结果 {index} ---")
        print(document.page_content)
        print("元数据：", document.metadata)