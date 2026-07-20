import re
import json
from concurrent.futures import ThreadPoolExecutor,as_completed
from typing import List

from langchain_core.messages import HumanMessage
from sentence_transformers import util

from config import llm, embedding_model
from vector_store import vector_retriever, bm25_retriever


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


# ===================== RRF 融合检索 =====================

def reciprocal_rank_fusion(*doc_lists, k: int = 60, top_n: int = 15) -> List[str]:
    """
    RRF (Reciprocal Rank Fusion) 融合多个检索器的排序结果

    公式：RRF_score(d) = Σ 1/(k + rank_i(d))
    rank_i(d) 从 1 开始计数

    Args:
        *doc_lists: 多个已排序的 Document 列表（每个按检索分数降序）
        k: RRF 平滑常数，默认 60
        top_n: 返回前 N 个 chunk 文本

    Returns:
        按 RRF score 降序排列的前 top_n 个 chunk 文本
    """
    rrf_scores = {}
    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list, start=1):
            chunk = doc.page_content
            rrf_scores[chunk] = rrf_scores.get(chunk, 0) + 1.0 / (k + rank)

    # 按 RRF score 降序排列
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in sorted_chunks[:top_n]]
# ===================== 向量加BM25并行函数=====================
def parallel_vec_BM25_retriever(query:str, top_n: int):
    #定义独立检索任务
    def vec_task():
        return vector_retriever.invoke(query)
    def bm25_task():
        return bm25_retriever.invoke(query)
    #开启线程池并发执行两个检索
    with ThreadPoolExecutor(max_workers=2) as executor:
        #提交两个任务
        futures_vec = executor.submit(vec_task)
        futures_bm25 = executor.submit(bm25_task)

        #设置阻塞获取两路结果
        vec_docs = futures_vec.result()
        bm25_docs = futures_bm25.result()

    return vec_docs, bm25_docs
# ===================== 检索函数 =====================

def my_rag_retrieve(query: str, top_n: int = 15) -> List[str]:
    """
    纯检索（不生成）：向量检索 + BM25 → RRF 融合 → 返回 top_n chunks
    用于子问题收集 chunk 场景
    """
    try:
        vec_docs, bm25_docs= parallel_vec_BM25_retriever(query, top_n=top_n)
        return reciprocal_rank_fusion(vec_docs,bm25_docs, k=60, top_n=top_n)
    except Exception as e:
        print(f"检索超时或失败: {e}")
        return []
# ===================== 问题分解 =====================

def decompose_query(query: str) -> dict:
    """
    LLM 判断复杂度并分解子问题

    Returns:
        {"is_complex": False, "sub_queries": [原始问题]}
        或 {"is_complex": True, "sub_queries": [子问题1, 子问题2, ...]}
    """
    prompt = f"""你是一个问题分析专家。判断以下问题是否需要拆分为多个子问题来分别检索回答。

    规则：
    1. 简单的事实查询、单一概念解释、单一操作步骤 → 不需要分解
    2. 涉及对比、多个方面、多个步骤、因果关系等复杂问题 → 需要分解为2-4个子问题
    3. 每个子问题应该是独立、简洁、可检索的查询语句
    
    用户问题：{query}
    
    请严格按以下JSON格式输出（不要输出任何其他内容，只输出JSON）：
    不分解：{{"is_complex": false, "sub_queries": ["原始问题原文"]}}
    需分解：{{"is_complex": true, "sub_queries": ["子问题1", "子问题2"]}}
    最多拆4个子问题！
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        # 尝试提取 JSON（防御 LLM 输出多余文字）
        content = response.content.strip()
        # 查找 JSON 部分
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # 校验格式
            if "sub_queries" in result and isinstance(result["sub_queries"], list):
                result.setdefault("is_complex", len(result["sub_queries"]) > 1)
                # 限制子问题数量
                result["sub_queries"] = result["sub_queries"][:4]
                return result
    except (json.JSONDecodeError, KeyError):
        pass

    # 解析失败，默认不分解
    return {"is_complex": False, "sub_queries": [query]}