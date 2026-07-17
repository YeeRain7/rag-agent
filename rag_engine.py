import re
from typing import List

from langchain_core.messages import HumanMessage
from sentence_transformers import util

from config import llm, embedding_model, cross_encoder
from vector_store import ensemble_retriever


# ===================== 意图路由：关键词列表 =====================

CHAT_KW = [
    "你好", "嗨", "hi", "哈喽", "在吗", "在不在", "谢谢", "感谢", "多谢",
    "再见", "拜拜", "回见", "哈哈", "哈哈哈", "嘿嘿", "笑死",
    "你是谁", "你叫什么", "你能干什么", "早上好", "下午好", "晚上好",
    "吃饭了吗", "最近怎么样", "有空吗", "最近", "忙吗", "hello", "fun"
]

KNOWLEDGE_KW = [
    "是什么", "什么是", "啥是", "怎么", "怎么样", "怎么做",
    "如何", "如何实现", "如何使用", "定义", "概念", "含义",
    "文档", "资料", "参考", "查询", "检索", "查找",
    "原理", "机制", "逻辑", "步骤", "流程", "操作",
    "方法", "方式", "方案", "教程", "指南", "实例",
    "区别", "对比", "优缺点", "原因", "为什么", "注意事项"
]


# ===================== 意图路由函数 =====================

def semantic_intent(query: str) -> str:
    """
    语义相似度二分类：返回 chat / knowledge / unknown
    """
    chat_prompt = "日常闲聊、打招呼、寒暄对话"
    knowledge_prompt = "知识查询、文档检索、概念问答"

    # 向量化
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
    response = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"max_tokens": 10}
    )
    raw_text = response.content.strip()
    # 正则提取目标关键词，抵御模型乱输出
    if re.search(r"\bknowledge\b", raw_text):
        return "knowledge"
    elif re.search(r"\bchat\b", raw_text):
        return "chat"
    else:
        # 无法识别默认走知识库查询，避免漏召回
        return "knowledge"


# ===================== RAG 核心链路 =====================

def my_rag_chain(query: str) -> str:
    """完整RAG链路：混合检索 → 重排 → LLM生成"""

    # 1. 混合检索
    def retrieve() -> List[str]:
        # 一次调用同时执行向量检索 + BM25关键词检索，自动融合结果
        retrieved_docs = ensemble_retriever.invoke(query)
        return [doc.page_content for doc in retrieved_docs]

    retrieved_chunks = retrieve()

    # 2. 重排
    def rerank(top_k: int = 5) -> List[str]:
        pairs = [(query, chunk) for chunk in retrieved_chunks]
        scores = cross_encoder.predict(pairs)
        chunk_with_score = list(zip(retrieved_chunks, scores))
        chunk_with_score.sort(key=lambda x: x[1], reverse=True)
        # 低分过滤
        valid = [(c, s) for c, s in chunk_with_score if s > 0.25]
        return [item[0] for item in valid[:top_k]]

    reranked_chunks = rerank()

    # 3. LLM 生成回答
    if not reranked_chunks:
        return "知识库暂无相关内容，请尝试换个问法或提供更多关键词。"

    prompt = f"""你是一个知识助手。请严格根据下面的【参考资料】回答用户的【问题】。
    如果资料中没有相关信息，直接说"资料中未提及相关内容"。
    禁止编造任何信息，可以适当引入资料以外的与内容相关的自身训练数据进行补充。

    【问题】
    {query}

    【参考资料】
    {"\n\n".join(reranked_chunks)}

    请基于上述资料作答："""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
