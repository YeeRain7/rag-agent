import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch

#LangGraph 相关依赖
from langgraph.graph import StateGraph, END, MessagesState
from dotenv import load_dotenv
from typing import List

# LangChain 相关导入
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# RAG 向量、重排相关导入
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

#文本预处理
import fitz  # PyMuPDF  文本加载--->pdf解析
import re      #文本清洗

# 物理核心8核
cpu_core_count = 8
os.environ["OMP_NUM_THREADS"] = str(cpu_core_count)
os.environ["MKL_NUM_THREADS"] = str(cpu_core_count)
torch.set_num_threads(cpu_core_count)
torch.set_num_interop_threads(2)   # interop线程保持少量，负责并行任务调度，不建议开大

# ===================== 全局一次性初始化（只执行1次） =====================
load_dotenv()
# 1. LLM 初始化
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("api_key"),
    base_url="https://api.deepseek.com/v1"
)
ds_client = OpenAI(
    api_key=os.getenv("api_key"),
    base_url="https://api.deepseek.com/v1"
)

# 2. 向量模型、重排模型全局加载
embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')


#3.0文本清洗函数
def clean_markdown(text: str) -> str:
    # 1. 移除图片（你的RAG看不了图，全删掉）
    # 匹配 ![alt](url) 或 <img> 标签
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'<img.*?>', '', text)

    # 2. 移除网页链接，只保留链接文字（可选）
    # 把 [文字](链接) 变成 文字
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # 3. 统一换行符：多个连续空行合并成两个换行（作为一个分块标记）
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # 4. 移除行尾空格和过长空白
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()

#3.1文本加载函数
def load_pdf(filepath):
    """从PDF中提取纯文本"""
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# 全局统一分块器（只初始化一次）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "，", " "]
)

# 4. 读取文档、分片
def load_all_docs(folder_path: str):
    all_chunks = []
    # 遍历文件夹所有文件
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        raw_text = ""
        # 分支读取不同格式文件
        if filename.endswith((".md", ".txt")):
            with open(filepath, 'r', encoding="utf-8") as f:
                raw_text = f.read()
        elif filename.endswith(".pdf"):
            raw_text = load_pdf(filepath)
        else:
            continue  # 跳过不支持的文件类型

        # 1. 清洗markdown标记
        clean_text = clean_markdown(raw_text)
        # 2. 替换原粗暴分割，使用语义分块
        chunks = text_splitter.split_text(clean_text)
        # 过滤空片段
        chunks = [c.strip() for c in chunks if c.strip()]
        # 加入全局列表
        all_chunks.extend(chunks)
        print(f"已加载 {filename}，得到 {len(chunks)} 个语义片段")

    # 全局去重
    all_chunks = list(dict.fromkeys(all_chunks))
    print(f"总计加载 {len(all_chunks)} 个有效片段")
    return all_chunks

# 5. 初始化内存向量库，存入所有分片
chromadb_client = chromadb.PersistentClient(path="./chroma_db_agent")
chromadb_collection = chromadb_client.get_or_create_collection(name="default")

#切片向量化函数
def embed_chunk(chunk: str) -> list[float]:
    embedding = embedding_model.encode(chunk)
    return embedding.tolist()

# 优化：判断集合是否为空，仅空库时才加载文档入库，避免重复插入
if chromadb_collection.count() == 0:
    print("向量库为空，正在加载文档并持久化...")
    chunks = load_all_docs("./knowledge_base")
    print(f"一共 {len(chunks)} 个片段，开始批量编码向量")
    # 批量向量化，比循环快很多
    embeddings = embedding_model.encode(chunks, batch_size=16).tolist()
    ids = [str(i) for i in range(len(chunks))]

    # 小批次写入！！每批100条，大幅降低sqlite压力
    batch_size = 100
    total = len(chunks)
    for start in range(0, total, batch_size):
        end = start + batch_size
        chromadb_collection.add(
            documents=chunks[start:end],
            embeddings=embeddings[start:end],
            ids=ids[start:end]
        )
        print(f"✅ 已写入 {min(end, total)} / {total}")
    print("🎉 全部入库完成！")
else:
    print(f"已加载持久化向量库，现有文档总数：{chromadb_collection.count()}")

# ===================== RAG 核心函数（仅检索+生成，无重复加载） =====================
def my_rag_chain(query: str):
    # 1. 向量召回
    def retrieve(top_k: int = 15) -> List[str]:
        query_embedding = embed_chunk(query)
        results = chromadb_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results['documents'][0]

    retrieved_chunks = retrieve()

    # 2. 重排
    def rerank(top_k: int = 5) -> List[str]:
        pairs = [(query, chunk) for chunk in retrieved_chunks]
        scores = cross_encoder.predict(pairs)
        chunk_with_score = list(zip(retrieved_chunks, scores))
        chunk_with_score.sort(key=lambda x: x[1], reverse=True)
        #低分过滤
        valid = [(c,s) for c, s in chunk_with_score if s>0.25]
        return [item[0] for item in valid[:top_k]]

    reranked_chunks = rerank()

    # 3. LLM 生成回答
    prompt = f"""请你作为一个知识助手，根据下面提供的【参考资料】回答用户的【问题】。
如果资料里没有相关信息，直接说“资料中未提及相关内容”，生成准确内容。

【问题】
{query}
【参考资料】
{"\n\n".join(reranked_chunks)}

请基于上述内容作答，模仿人类的助手说话，不要编造信息。"""

    response = ds_client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content


# ===================== 工具、Agent 初始化 =====================
@tool
def search_my_knowledge(query: str):
    """搜索本地私有知识库，仅当用户问题需要查阅文档时调用此工具
    Args:
        query: 用户原始提问
    """
    return my_rag_chain(query)

tools = [search_my_knowledge]

# 创建标准工具Agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个专业的知识助手，可以调用知识库工具回答用户问题，工具返回结果后整理成自然回答给用户",
)

class ChatState(MessagesState):
    query: str
    answer: str
def chatbot_node(state: ChatState):
    result = agent.invoke({"messages":state["messages"]})
    final_msg = result["messages"][-1]
    return {
        "messages": [final_msg],
        "answer": final_msg.content
    }

graph = StateGraph(ChatState)
graph.add_node("chatbot",chatbot_node)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()
# ===================== 调用测试 =====================
if __name__ == "__main__":
    # 初始化全局对话状态，存放完整历史消息
    current_state = {
        "messages": [],
        "query": "",
        "answer": ""
    }
    print("对话已启动，输入「退出」结束对话\n")

    while True:
        # 接收用户输入
        user_input = input("用户：")
        # 退出判断
        if user_input.strip() == "退出":
            print("对话结束")
            break

        # 更新当前状态：写入本轮用户提问
        current_state["query"] = user_input
        current_state["messages"].append(HumanMessage(content=user_input))

        # 调用LangGraph执行流程
        res = app.invoke(current_state)

        # 打印AI回复
        print(f"AI：{res['answer']}\n")

        # 刷新状态，保留完整对话历史，供下一轮上下文使用
        current_state = res