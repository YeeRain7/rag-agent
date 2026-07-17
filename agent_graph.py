from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage

from config import llm
from rag_engine import my_rag_chain, semantic_intent, llm_router, CHAT_KW, KNOWLEDGE_KW


# ===================== Tool 定义与 Agent 创建 =====================

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

# ===================== 状态定义 =====================

class ChatState(MessagesState):
    query: str
    answer: str
    search_hint: str | None  # 仅用于本轮检索临时提示


# ===================== 图节点 =====================

def agent_node(state: ChatState):
    """核心Agent节点：调用LangChain Agent执行RAG查询"""
    messages = state["messages"]
    search_hint = state.get("search_hint")

    # 如果存在反思生成的检索优化词，追加提示给Agent
    if search_hint:
        hint_msg = SystemMessage(content=f"知识库检索优先使用关键词：{search_hint}")
        invoke_messages = messages + [hint_msg]
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
