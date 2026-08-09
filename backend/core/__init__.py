"""
core 包 — RAG 智能问答系统核心模块。

所有重量级资源（向量库、检索器、LangGraph Agent）通过 init() 函数延迟初始化，
由 FastAPI 的 lifespan startup 事件调用，避免 import 时自动加载模型。
"""

# ===================== 延迟加载单例 =====================

vector_retriever = None
bm25_retriever = None
chromadb_collection = None
chromadb_client = None
vector_store = None
langgraph_app = None


def init(knowledge_base_path: str = "./knowledge_base"):
    """
    初始化所有核心组件（向量库 + 检索器 + LangGraph Agent）。

    由 FastAPI startup 事件调用。可重复调用（幂等：仅首次执行实际初始化）。

    Args:
        knowledge_base_path: 知识库文档目录路径
    """
    global vector_retriever, bm25_retriever, chromadb_collection
    global chromadb_client, vector_store, langgraph_app

    if langgraph_app is not None:
        print("核心组件已初始化，跳过重复初始化")
        return

    # 1. 初始化向量库和检索器
    from .vector_store import init_vector_store
    result = init_vector_store(knowledge_base_path)

    vector_retriever = result["vector_retriever"]
    bm25_retriever = result["bm25_retriever"]
    chromadb_collection = result["chromadb_collection"]
    chromadb_client = result["chromadb_client"]
    vector_store = result["vector_store"]

    # 2. 同步更新 agent_graph 模块级变量（供内部引用）
    from . import agent_graph
    agent_graph.vector_retriever = vector_retriever
    agent_graph.bm25_retriever = bm25_retriever

    # 3. 构建 LangGraph Agent
    from .agent_graph import build_graph
    langgraph_app = build_graph()
    agent_graph.app = langgraph_app

    print("[OK] 所有核心组件初始化完成")
