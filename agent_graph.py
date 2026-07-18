from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage

from config import llm, cross_encoder
from rag_engine import (
    my_rag_retrieve,
    decompose_query,
    semantic_intent, llm_router,
    CHAT_KW, KNOWLEDGE_KW
)


# ===================== Tool 定义 =====================

@tool
def search_knowledge_tool(query: str) -> str:
    """搜索本地知识库。对每个（子）问题调用此工具获取相关文档片段。
    返回经过RRF融合和Cross-Encoder重排后的top 5文档片段。
    用法: search_knowledge("你要搜索的问题")"""
    chunks = my_rag_retrieve(query, top_n=15)
    if not chunks:
        return "[未找到相关内容，请如实告知用户知识库暂无此信息]"

    # Cross-Encoder 重排 → top 5
    pairs = [(query, c) for c in chunks]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top = [c for c, s in ranked if s > 0.25][:5]

    if not top:
        return "[未找到相关内容，请告知用户知识库暂无此信息]"

    return "\n---\n".join(f"[来源{i}] {c}" for i, c in enumerate(top, 1))


@tool
def decompose_question_tool(question: str) -> str:
    """判断问题复杂度并拆分。对于涉及对比、多个方面、多步骤的复杂问题，
    拆分为2-4个独立的子问题。简单问题返回无需拆分。
    用法: decompose_question("用户的问题")"""
    result = decompose_query(question)
    if result.get("is_complex") and len(result.get("sub_queries", [])) > 1:
        lines = [f"{i}. {q}" for i, q in enumerate(result["sub_queries"], 1)]
        return "拆分为以下子问题：\n" + "\n".join(lines)
    return "该问题无需拆分，直接搜索即可。"


# ===================== Agent 创建 =====================

tools = [search_knowledge_tool, decompose_question_tool]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""你是一个专业的知识助手，通过工具调用来回答用户问题。

工具使用规则：
1. 先判断问题类型：
   - 涉及"对比"、"区别"、"优缺点"、"分别"、多个方面、多步骤推理 → 先调用 decompose_question 拆分
   - 简单概念解释、单一事实查询 → 直接调用 search_knowledge
2. 拆分后，对每个子问题分别调用 search_knowledge
3. 综合所有检索到的文档片段，给出结构清晰、内容准确的回答
4. 严格基于检索结果回答，检索不到的内容诚实告知用户
5. 回答最后列出所引用的来源编号
"""
)


# ===================== 状态定义 =====================

class ChatState(MessagesState):
    query: str
    answer: str
    search_hint: str | None  # 仅用于本轮检索临时提示
    sub_queries: list[str] | None  # 记录分解后的子问题（调试用）


# ===================== 图节点 =====================

def agent_node(state: ChatState):
    """核心Agent节点：LLM自主决策工具调用（分解/搜索/合成）"""
    messages = state["messages"]
    search_hint = state.get("search_hint")

    # 如果存在反思生成的检索优化词，注入Agent
    if search_hint:
        hint_msg = SystemMessage(content=f"检索优化提示：{search_hint}")
        invoke_messages = [hint_msg] + list(messages)
    else:
        invoke_messages = messages

    result = agent.invoke({"messages": invoke_messages})
    final_msg = result["messages"][-1]
    return {
        "messages": [final_msg],
        "answer": final_msg.content,
        "search_hint": None
    }


def chatbot_node(state: ChatState):
    """闲聊节点：直接LLM对话"""
    prompt = SystemMessage(content="你是友好对话助手，正常闲聊")
    resp = llm.invoke([prompt] + state["messages"])
    return {"messages": [resp], "answer": resp.content}


def router(state: ChatState):
    """路由节点：三层混合仲裁（关键词 + 语义相似度 + LLM兜底）"""
    # 新用户请求进入分支前，强制清理上一轮残留检索提示
    state["search_hint"] = None
    query = state["query"]

    # ========= 第一级：规则快筛层 =========
    # 方法一：关键词匹配
    is_chat_kw = any(kw in query for kw in CHAT_KW)
    is_knowledge_kw = any(kw in query for kw in KNOWLEDGE_KW)

    # 方法二：语义相似度匹配
    intent_semantic = semantic_intent(query)

    # ========= 第二级：仲裁逻辑 =========
    # 冲突处理前提：如果两种关键字都包含，直接给大模型兜底，跳过语义分析函数
    if is_chat_kw and is_knowledge_kw:
        is_chat_kw = False
        is_knowledge_kw = False

    # 情况1：如果方法一和方法二都指向A，那就走A
    if is_knowledge_kw and intent_semantic == "knowledge":
        return "agent_node"
    if is_chat_kw and intent_semantic == "chat":
        return "chatbot_node"

    # 情况2：出现冲突，交给LLM仲裁
    llm_intent = llm_router(query)
    if llm_intent == "knowledge":
        return "agent_node"
    else:
        return "chatbot_node"


def reflection_node(state: ChatState):
    """反思评估节点：评估回答质量，不足时触发重试"""
    query = state["query"]
    answer = state["answer"]

    reflection_prompt = f"""
    用户问题:{query}
    系统回答:{answer}

    评估规则：
    1. 判断回答是否准确、完整回答用户问题。
    2. 对于定义、概念类问题，追求清晰准确就行，不需要重试。
    3. 如果回答信息不足、内容错误，输出格式：
    `需要重试|优化检索关键词：xxx`
    4. 如果回答合格，直接输出：`不需要重试`

    示例：
    需要重试|优化检索关键词：RAG技术原理
    不需要重试
    只允许使用以上两种格式输出！
    """
    response = llm.invoke(reflection_prompt)
    content = response.content.strip()

    if content.startswith("需要重试|"):
        # 拆分得到优化后的检索关键词
        _, new_search_query = content.split("|")
        return {
            "search_hint": new_search_query,
            "need_retry": True,
            "retry_count": state.get("retry_count", 0) + 1
        }
    else:
        return {"need_retry": False, "final_answer": answer}


def should_retry(state: ChatState):
    """反思条件路由：最多重试2次"""
    if state.get("need_retry") and state.get("retry_count", 0) < 2:
        return "agent_node"
    else:
        return END


# ===================== 图构建 =====================

graph = StateGraph(ChatState)
graph.add_node("agent", agent_node)
graph.add_node("router", router)
graph.add_node("chatbot", chatbot_node)
graph.add_node("reflection", reflection_node)

graph.add_conditional_edges(
    "__start__",
    router,
    {
        "agent_node": "agent",
        "chatbot_node": "chatbot"
    }
)
graph.add_edge("agent", "reflection")
graph.add_conditional_edges(
    "reflection",
    should_retry,
    {
        "agent_node": "agent",
        END: END
    }
)
graph.add_edge("chatbot", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)
