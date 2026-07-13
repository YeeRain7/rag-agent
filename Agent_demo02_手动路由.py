import os

from langgraph import graph

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

#LangGraph 相关依赖
from langgraph.graph import StateGraph, END, MessagesState

from dotenv import load_dotenv
from typing import List

# LangChain 相关导入
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# RAG 向量、重排相关导入
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

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

# 3. 读取文档、分片
def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, 'r', encoding="utf-8") as file:
        content = file.read()
    return [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]


chunks = split_into_chunks("doc.md")
chunks = list(dict.fromkeys(chunks)) # 去除完全重复片段

# 4. 初始化内存向量库，存入所有分片
chromadb_client = chromadb.PersistentClient(path="./chroma_db_demo02")
chromadb_collection = chromadb_client.get_or_create_collection(name="default")

#切片向量化函数
def embed_chunk(chunk: str) -> list[float]:
    embedding = embedding_model.encode(chunk)
    return embedding.tolist()

# 优化：判断集合是否为空，仅空库时才加载文档入库，避免重复插入
if chromadb_collection.count() == 0:
    print("向量库为空，正在加载文档并持久化...")
    chunks = split_into_chunks("doc.md")
    embeddings = [embed_chunk(chunk) for chunk in chunks]
    ids = [str(i) for i in range(len(chunks))]
    chromadb_collection.add(documents=chunks, embeddings=embeddings, ids=ids)
else:
    print(f"已加载持久化向量库，现有文档总数：{chromadb_collection.count()}")


# ===================== RAG 核心函数（仅检索+生成，无重复加载） =====================
def my_rag_chain(query: str):
    # 1. 向量召回
    def retrieve(top_k: int = 5) -> List[str]:
        query_embedding = embed_chunk(query)
        results = chromadb_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results['documents'][0]

    retrieved_chunks = retrieve()

    # 2. 重排
    def rerank(top_k: int = 3) -> List[str]:
        pairs = [(query, chunk) for chunk in retrieved_chunks]
        scores = cross_encoder.predict(pairs)
        chunk_with_score = list(zip(retrieved_chunks, scores))
        chunk_with_score.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in chunk_with_score[:top_k]]

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
def search_my_knowledge(query: str):
    """搜索本地私有知识库，仅当用户问题需要查阅文档时调用此工具
    Args:
        query: 用户原始提问
    """
    return my_rag_chain(query)

class ChatState(MessagesState):
    query: str
    answer: str
def respond_node(state):
    # 简单直接回复
    answer = f"你好，关于「{state['query']}」这个问题，我可以直接回答……"
    state["answer"] = answer
    return state
def retrieve_node(state):
    state["answer"] = search_my_knowledge(state["query"])
    answer = state["answer"]
    return answer
def router(state):
    # 简单判断：问题里有没有关键词需要检索
    if any(kw in state["query"] for kw in ["是什么", "怎么", "定义"]):
        return "retrieve"
    else:
        return "respond"

graph = StateGraph(ChatState)
graph.add_node("router",router)
graph.add_node("retrieve",retrieve_node)
graph.add_node("respond",respond_node)

graph.add_conditional_edges(
    "router",
    router,
    {
        "retrieve": "retrieve",
        "respond": "respond"
    }
)

graph.add_edge("retrieve", END)
graph.add_edge("respond", END)

app = graph.compile()
# ===================== 调用测试 =====================
if __name__ == "__main__":
    # response = agent.invoke({
    #     "messages": [HumanMessage(content="哆啦A梦使用的3个秘密道具分别是什么?")]
    # })
    # print("问题："+response["messages"][0].content)
    # print("回答："+response["messages"][-1].content)
    # 打印完整对话结果
    # for msg in response["messages"]:
    # print(f"\n【{type(msg).__name__}】{msg.content}")
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