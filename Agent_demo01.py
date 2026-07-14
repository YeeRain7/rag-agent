import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch

#LangGraph 相关依赖
from langgraph.graph import StateGraph, END, MessagesState,START
from dotenv import load_dotenv
from typing import List

# LangChain 相关导入
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# RAG 向量、重排相关导入
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder,util

#文本预处理
import fitz    # PyMuPDF  文本加载--->pdf解析
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

#意图识别函数--->余弦相似度
def semantic_intent(query: str) -> str:
    """
    语义相似度二分类：返回 chat / knowledge
    """
    # 定义两类意图标准句
    chat_prompt = "日常闲聊、打招呼、寒暄对话"
    knowledge_prompt = "知识查询、文档检索、概念问答"

    # 向量化，必须传入列表
    q_emb = embedding_model.encode([query], convert_to_tensor=True)
    chat_emb = embedding_model.encode([chat_prompt], convert_to_tensor=True)
    knowledge_emb = embedding_model.encode([knowledge_prompt], convert_to_tensor=True)

    # 计算相似度
    score_chat = util.cos_sim(q_emb, chat_emb).item()
    score_know = util.cos_sim(q_emb, knowledge_emb).item()

    threshold = 0.02
    if score_know - score_chat > threshold:
        return "knowledge"
    elif score_chat - score_know > threshold:
        return "chat"
    else:
        # 分数接近，留给LLM仲裁
        return "unknown"
#意图识别函数--->LLM
def llm_router(query: str) -> str:
    """
       LLM兜底意图仲裁
       返回值： "chat" 闲聊 | "knowledge" 知识库问答
       """
    prompt = f"""你是一个人类语言情感分析专家
    请判断用户提问属于【闲聊对话】还是【知识库知识查询】
    规则：
    1. 闲聊chat：打招呼、感慨、日常聊天、情绪抒发，不需要查询文档资料
    2. 知识查询knowledge：询问定义、原理、步骤、文档相关内容，需要检索知识库
    只允许输出单词，严格只能输出 chat 或者 knowledge，不要额外文字！

    用户问题：{query}
    结果：
    """
    resp = ds_client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": prompt},
            {"role":"user", "content": query}
            ],
        temperature=0.0,
        max_tokens=10
    )
    raw_text = resp.choices[0].message.content
    # 正则提取目标关键词，抵御模型乱输出
    if re.search(r"\bknowledge\b", raw_text):
        return "knowledge"
    elif re.search(r"\bchat\b", raw_text):
        return "chat"
    else:
        # 无法识别默认走知识库查询，避免漏召回
        return "knowledge"

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
如果资料里没有相关信息，直接说“资料中未提及相关内容”，生成与【问题】相关的准确内容。
    强制要求：
    1. ✅ **先检索知识库**：收到问题后，第一时间调用搜索工具查找相关内容
    2. ✅ **再整理回答**：基于知识库返回的结果，并整理成自然语言回复
    3. ✅ **知识库无结果时**：再依赖自身知识补充回答，并说明详细情况  

【问题】
{query}
【参考资料】
{"\n\n".join(reranked_chunks)}

请基于上述内容作答，模仿人类的助手说话，语义清晰连贯，有适当分点，不要编造信息。"""

    response = ds_client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
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
    system_prompt="你是一个专业的知识助手，必须优先调用知识库工具回答用户问题，工具返回结果后整理成自然回答给用户，如果知识库没有该问题的相关内容，才利用自身训练数据回答",
)

class ChatState(MessagesState):
    query: str
    answer: str
def agent_node(state: ChatState):
    result = agent.invoke({"messages":state["messages"]})
    final_msg = result["messages"][-1]
    return {
        "messages": [final_msg],
        "answer": final_msg.content
    }

def chatbot_node(state: ChatState):
    response = app.invoke({"messages":state["messages"]})
    return {
        "messages": [response],
        "answer": response.content
    }

def router(state:ChatState):  #路由节点----->混合路由仲裁
    query = state["query"]
    # ========= 第一级：规则快筛层 =========
    # 仲裁规则：方法一和方法二只要有一个明确判断，就采纳；若两者结论相反，触发兜底。

    # 方法一：关键词匹配
    chat_kw = [
        "你好", "嗨", "hi", "哈喽", "在吗", "在不在","谢谢", "感谢", "多谢",
        "再见", "拜拜", "回见","哈哈", "哈哈哈", "嘿嘿", "笑死",
        "你是谁", "你叫什么", "你能干什么","早上好", "下午好", "晚上好",
        "吃饭了吗", "最近怎么样", "有空吗","最近","忙吗","hello","fun"
    ]
    knowledge_kw = [
        "是什么", "什么是", "啥是","怎么", "怎么样", "怎么做",
        "如何", "如何实现", "如何使用","定义", "概念", "含义",
        "文档", "资料", "参考","查询", "检索", "查找",
        "原理", "机制", "逻辑","步骤", "流程", "操作",
        "方法", "方式", "方案","教程", "指南", "实例",
        "区别", "对比", "优缺点","原因", "为什么", "注意事项"
    ]

    is_chat_kw = any(kw in query for kw in chat_kw)
    is_knowledge_kw = any(kw in query for kw in knowledge_kw)
    # 方法二：语义相似度匹配（用你本地的轻量模型）
    # 利用语义判断函数semantic_intent(query) 返回 "chat" 或 "knowledge"
    intent_semantic = semantic_intent(query)

    # ========= 第二级：仲裁逻辑 =========
    #冲突处理前提:如果两种关键字都包含,直接给大模型兜底,跳过语义分析函数
    if is_chat_kw and is_knowledge_kw:
        is_chat_kw =False
        is_knowledge_kw = False
    # 情况1. 如果方法一和方法二都指向A，那就走A
    if is_knowledge_kw and intent_semantic =="knowledge":
        return "agent_node"
    if is_chat_kw and intent_semantic=="chat":
        return "chatbot_node"
    # 情况2：出现冲突，交给LLM仲裁
    llm_intent = llm_router(query)
    if llm_intent == "knowledge":
        return "agent_node"
    else:
        return "chatbot_node"

graph = StateGraph(ChatState)
graph.add_node("agent",agent_node)
graph.add_node("router",router)
graph.add_node("chatbot",chatbot_node)

graph.add_conditional_edges(
    "__start__",
    router,
    {
        "agent_node":"agent",
        "chatbot_node":"chatbot"
    }
)
graph.add_edge("agent",END)
graph.add_edge("chatbot",END)

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