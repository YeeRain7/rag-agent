"""
FastAPI 应用入口 — RAG 智能问答系统

启动时触发向量库初始化（替代 vector_store.py 的模块级自动加载），
通过 lifespan 管理资源生命周期，LangGraph StateGraph 作为全局单例共享。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import init as core_init
from api.chat import router as chat_router
from api.knowledge import router as knowledge_router


# ===================== 生命周期管理 =====================

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    应用生命周期：
    - startup:  初始化向量库 + 检索器 + LangGraph Agent（避免 import 就加载模型）
    - shutdown: 清理资源（ChromaDB 客户端等）
    """
    # ===== STARTUP =====
    print("[STARTUP] 正在初始化 RAG 核心组件...")
    core_init("./knowledge_base")
    print("[STARTUP] RAG 核心组件就绪，开始接受请求")

    yield  # 应用运行中

    # ===== SHUTDOWN =====
    print("[SHUTDOWN] 正在清理资源...")
    from core import chromadb_client
    if chromadb_client is not None:
        try:
            # ChromaDB PersistentClient 没有显式 close，依赖 GC
            pass
        except Exception:
            pass
    print("[SHUTDOWN] 资源清理完成")


# ===================== 应用实例 =====================

app = FastAPI(
    title="RAG 智能问答系统",
    description="基于 LangGraph + ChromaDB + RRF 融合的智能文档问答系统",
    version="1.0.0",
    lifespan=lifespan,
)


# ===================== CORS 中间件 =====================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite 开发服务器
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== 路由注册 =====================

app.include_router(chat_router, prefix="/ws", tags=["聊天"])
app.include_router(knowledge_router, prefix="/api/knowledge", tags=["知识库"])


# ===================== 健康检查 =====================

@app.get("/health", tags=["系统"])
async def health():
    """健康检查端点（可用于 K8s liveness probe）"""
    return {"status": "ok"}
