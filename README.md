# LangChain Assistant

一个用于学习 LangChain、Tool、Agent、Memory 和 RAG 的命令行智能助手项目。

项目使用 DeepSeek 生成回答，使用本地 Ollama Embedding 模型生成文本向量，并通过 Chroma 检索本地知识库。

## 功能

- 通过 OpenAI 兼容接口调用 DeepSeek。
- 使用 LangChain Agent 自动选择并调用工具。
- 使用计算工具计算商品总价。
- 使用 LangGraph `InMemorySaver` 保存当前进程中的短期对话记忆。
- 使用 Ollama `qwen3-embedding:0.6b` 生成文本向量。
- 使用 Chroma 持久化保存和检索本地知识库。
- 使用 Agentic RAG 回答训练营手册相关问题。

## 运行流程

```text
用户输入问题
→ LangChain Agent 判断如何处理
├── 商品总价问题 → calculate_total
├── 训练营问题   → search_course_handbook
└── 普通问题     → DeepSeek 直接回答
```

知识库问答流程：

```text
用户问题
→ Agent 调用知识库工具
→ Ollama 把问题转换成向量
→ Chroma 搜索相似文本块
→ 工具把检索结果交给 Agent
→ DeepSeek 根据检索结果生成回答
```

## 技术栈

- Python 3.12
- uv
- LangChain
- LangGraph
- DeepSeek
- Ollama
- Qwen3 Embedding
- Chroma

## 项目结构

```text
langchain-assistant/
├── app/
│   ├── __init__.py
│   ├── knowledge.py       # 知识库读取、分块、索引和检索
│   ├── main.py            # 命令行程序入口
│   ├── model.py           # DeepSeek 模型初始化
│   └── tools.py           # 计算工具和知识库工具
├── data/
│   └── course_handbook.md # 原始知识库
├── .env.example           # 环境变量配置模板
├── .gitignore
├── pyproject.toml         # 项目配置和直接依赖
├── README.md
└── uv.lock                # 完整依赖锁文件
```

运行项目后还会生成：

```text
chroma_db/
```

这是根据原始知识库生成的本地向量数据库，不提交到 Git。

## 环境要求

运行项目前需要安装：

- Python 3.12 或更高版本
- uv
- Ollama

本项目使用的 Embedding 模型是：

```text
qwen3-embedding:0.6b
```

## 1. 克隆项目

```bash
git clone git@github.com:leafsilver/langchain-assistant.git
cd langchain-assistant
```

## 2. 安装项目依赖

```bash
uv sync --locked
```

`uv sync` 根据 `pyproject.toml` 和 `uv.lock` 创建或更新项目的 `.venv`，并安装所需依赖。

`--locked` 表示必须按照现有锁文件安装。如果 `pyproject.toml` 与 `uv.lock` 不一致，命令会报错，而不是自动修改锁文件。

## 3. 配置环境变量

复制配置模板：

```bash
cp .env.example .env
```

编辑 `.env`：

```ini
DEEPSEEK_API_KEY=YOUR_DEEPSEEK_API_KEY_HERE
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

变量说明：

- `DEEPSEEK_API_KEY`：DeepSeek API Key。
- `DEEPSEEK_BASE_URL`：DeepSeek 的 OpenAI 兼容接口地址。
- `DEEPSEEK_MODEL`：需要调用的模型名称。

真实的 API Key 只能保存在本地 `.env` 中，不要提交到 Git，也不要写入源码、README 或测试文件。

## 4. 准备 Ollama Embedding 模型

确保 Ollama 服务已经启动，然后下载模型：

```powershell
ollama pull qwen3-embedding:0.6b
```

查看已经安装的模型：

```powershell
ollama list
```

预期能够看到：

```text
qwen3-embedding:0.6b
```

项目通过以下地址访问 Ollama：

```text
http://127.0.0.1:11434
```

如果 Ollama 运行在 Windows，而项目运行在 WSL 中，需要确保 WSL 能够访问 Windows 的 localhost。

可以在 WSL 中测试：

```bash
curl http://127.0.0.1:11434/api/tags
```

能够返回 Ollama 模型列表，说明连接正常。

## 5. 建立知识库索引

第一次运行项目前，需要根据原始知识库建立 Chroma 向量索引：

```bash
uv run python -m app.knowledge
```

该命令会执行：

```text
读取 data/course_handbook.md
→ 分割成多个 Document
→ 调用 Ollama 生成向量
→ 写入 chroma_db
→ 执行一次示例相似度搜索
```

修改 `data/course_handbook.md` 后，需要重新运行该命令更新索引。

## 6. 运行智能助手

```bash
uv run python -m app.main
```

程序启动后可以连续输入问题：

```text
智能助手已启动，输入 exit 结束对话。

你：本期训练营的代号是什么？

助手：
本期训练营的代号是 LEAF-2026。
```

输入：

```text
exit
```

结束程序。

## 示例问题

查询知识库：

```text
本期训练营的代号是什么？
每周什么时候进行线下答疑？
作业最晚什么时候提交？
设备出现故障后可以自行拆机吗？
获得结业证明需要完成什么？
```

使用计算工具：

```text
一件商品15.5元，购买4件，一共多少钱？
```

测试短期记忆：

```text
我叫小叶，请记住我的名字。
我叫什么名字？
```

## 知识库实现

原始知识库位于：

```text
data/course_handbook.md
```

项目使用 `RecursiveCharacterTextSplitter` 分割文本：

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)
```

使用本地 Ollama 生成向量：

```python
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://127.0.0.1:11434",
)
```

使用 Chroma 持久化保存向量：

```python
vector_store = Chroma(
    collection_name="course_handbook",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)
```

## 工具

当前 Agent 拥有两个工具。

### `calculate_total`

根据商品单价和数量计算总价。

```text
单价 + 数量
→ 计算工具
→ 返回总价
```

### `search_course_handbook`

检索训练营手册。

```text
训练营问题
→ Ollama Embedding
→ Chroma 相似度搜索
→ 返回相关知识块
```

工具的函数名、参数类型和 Docstring 会提供给模型，帮助 Agent 判断什么时候调用工具。

## 短期记忆

项目使用：

```python
InMemorySaver
```

保存当前对话的消息状态。

相同程序运行期间，Agent 可以记住之前的对话；退出程序后，内存中的记录会消失。

当前记忆适合学习和本地测试，不是持久化存储。

## Git 忽略规则

以下内容不会提交到 Git：

```gitignore
.venv/
__pycache__/
*.pyc
.env*
chroma_db/
```

其中：

- `.venv/`：项目虚拟环境，可以根据锁文件重新创建。
- `__pycache__/` 和 `*.pyc`：Python 运行时缓存。
- `.env`：包含本地敏感配置。
- `chroma_db/`：可以根据原始知识库重新生成的向量数据库。

`.env.example` 是配置模板，需要提交到 Git，因此使用：

```gitignore
!.env.example
```

## 当前限制

- 短期记忆只保存在当前 Python 进程中。
- Chroma 使用本地持久化目录，适合学习和本地开发。
- 当前只读取一个 Markdown 知识库文件。
- 修改知识库后需要手动重新建立索引。
- 项目依赖 Windows Ollama 或其他能够通过配置地址访问的 Ollama 服务。
