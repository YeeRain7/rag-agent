# Vue + FastAPI 搭建 RAG 智能问答系统 — 实施计划

## Context

将现有的命令行 RAG 智能问答系统（6 模块、LangGraph + ChromaDB + RRF 融合）升级为 Web 应用。前端用 Vue 3，后端用 FastAPI，通过 WebSocket 实现流式对话。核心业务逻辑（`agent_graph.py`、`rag_engine.py`、`vector_store.py`、`document_loader.py`）保持不变，FastAPI 仅替换 `main.py` 的 CLI 对话循环。

---

## 1. 后端：FastAPI 改造

### 1.1 项目结构

```
backend/
├── main.py                  # FastAPI 应用入口 + 生命周期
├── api/
│   ├── __init__.py
│   ├── chat.py              # WebSocket 聊天端点
│   └── knowledge.py         # 知识库管理 REST 端点
├── core/
│   ├── __init__.py
│   ├── config.py            # 现有 config.py（基本不动）
│   ├── agent_graph.py       # 现有 agent_graph.py（不动）
│   ├── rag_engine.py        # 现有 rag_engine.py（不动）
│   ├── vector_store.py      # 现有 vector_store.py（微调：延迟初始化）
│   └── document_loader.py   # 现有 document_loader.py（不动）
├── models/
│   ├── __init__.py
│   └── schemas.py           # Pydantic 请求/响应模型
└── requirements.txt         # 新增依赖文件
```

### 1.2 关键改动点

#### A. `backend/main.py` — FastAPI 入口（新建）

```python
# 核心逻辑：
# - @app.on_event("startup"): 触发向量库初始化（原 vector_store.py 的模块级代码）
# - @app.on_event("shutdown"): 清理资源
# - 注册路由: chat.router, knowledge.router
# - CORS 中间件（允许前端跨域）
```

- 用 `@app.on_event("startup")` 替代 `vector_store.py` 的模块级初始化，避免 import 就加载模型
- 配置 CORS，允许 `localhost:5173`（Vite 开发服务器）
- 把 LangGraph `app`（StateGraph）作为全局单例，所有 WebSocket 连接共享

#### B. `backend/api/chat.py` — WebSocket 聊天端点（新建）

**核心挑战**：LangGraph 的 `astream_events` 可以流式输出 Agent 的每一步（tool 调用、LLM 生成 token），但 DeepSeek API 通过 LangChain 包装后是否支持逐 token 流需要验证。

**两种方案**：

| 方案 | 描述 | 适用场景 |
|---|---|---|
| **A. SSE 伪流式** | 用 `asyncio.to_thread` 跑 `app.invoke()`，完成后一次性推送给前端，前端模拟打字效果 | DeepSeek Flash 不支持 token 级流式时使用 |
| **B. astream_events 真流式** | `async for event in app.astream_events(...)` 逐 token 推 WebSocket | DeepSeek 支持时首选 |

**推荐先实现方案 A**（稳妥），后续再升级方案 B。

WebSocket 消息协议：
```json
// 客户端 → 服务端
{"type": "query", "session_id": "xxx", "content": "用户问题"}

// 服务端 → 客户端
{"type": "status", "stage": "routing"}           // 正在路由
{"type": "status", "stage": "retrieving"}         // 正在检索
{"type": "status", "stage": "generating"}          // 正在生成
{"type": "chunk", "content": "回答片段"}           // 流式内容（方案B）
{"type": "answer", "content": "完整回答",          // 最终结果
 "sources": [{"index": 1, "text": "..."}]}
{"type": "error", "message": "错误信息"}
```

- 每个 WebSocket 连接对应一个 LangGraph `thread_id`（用 `session_id` 映射）
- 用 `asyncio.to_thread` 包裹 `app.invoke()`，避免阻塞事件循环
- LangGraph 的 `config["configurable"]["thread_id"]` 用 session_id，实现多会话隔离

#### C. `backend/api/knowledge.py` — 知识库管理（新建）

REST 端点：

| 方法 | 路径 | 功能 |
|---|---|---|
| `GET` | `/api/knowledge/documents` | 列出已入库文档 |
| `POST` | `/api/knowledge/upload` | 上传 PDF/MD/TXT，触发分块+向量化 |
| `DELETE` | `/api/knowledge/{doc_id}` | 删除文档及对应向量 |
| `GET` | `/api/knowledge/stats` | 知识库统计（文档数、chunk 数） |

**上传流程**：
```
用户上传文件 → 保存到 knowledge_base/ → load_single_file() 分块
→ embedding_model.encode() 向量化 → chromadb_collection.add() 写入
→ 更新 BM25Retriever（需重建）
```

**注意事项**：
- 上传时加写入锁（`asyncio.Lock`），ChromaDB SQLite 不支持并发写
- 上传新文档后需要重建 BM25 索引（`BM25Retriever.from_texts(all_chunks)`）
- 大文件（如 61MB 的 PDF）可能需要较长时间，返回进度或异步任务

#### D. `backend/models/schemas.py` — Pydantic 模型（新建）

```python
from pydantic import BaseModel

class DocumentInfo(BaseModel):
    id: str
    filename: str
    chunk_count: int
    uploaded_at: str

class KnowledgeStats(BaseModel):
    document_count: int
    total_chunks: int

class UploadResponse(BaseModel):
    success: bool
    filename: str
    chunk_count: int
    message: str
```

#### E. `backend/core/vector_store.py` — 微调

需要改成**函数式初始化**而非模块级自动执行。把原来的模块级代码提取为 `init_vector_store()` 函数：

```python
# 原来：模块 import 时自动执行
# 改为：
def init_vector_store(knowledge_base_path: str = "./knowledge_base"):
    """由 FastAPI startup 事件调用"""
    # ... 原来的初始化逻辑 ...
    return vector_retriever, bm25_retriever, chromadb_collection
```

`rag_engine.py` 的 `from vector_store import vector_retriever, bm25_retriever` 需要调整为从函数参数注入，或者用一个全局单例在 startup 后设置好。

**最简单的做法**：在 `backend/core/__init__.py` 里保持单例，startup 时调用 `init()` 设置。

#### F. `backend/requirements.txt`（新建）

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
websockets>=12.0
python-multipart>=0.0.9       # 文件上传
langgraph
langchain
langchain-openai
langchain-community
langchain-chroma
langchain-text-splitters
chromadb
sentence-transformers
pymupdf
python-dotenv
```

### 1.3 对话 Session 持久化

当前用 `MemorySaver`（内存），服务重启丢失。改为 `SqliteSaver`：

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 在 agent_graph.py 中
memory = SqliteSaver.from_conn_string("checkpoints.db")
app = graph.compile(checkpointer=memory)
```

前端通过 `session_id` 参数切换/恢复会话。

---

## 2. 前端：Vue 3 搭建

### 2.1 技术选型

| 层面 | 选择 | 理由 |
|---|---|---|
| 脚手架 | Vite + Vue 3 + TypeScript | 最快冷启动，Vue 3 Composition API |
| UI 组件库 | **无**（手写）或 Element Plus | 手写更灵活，Element Plus 快出原型 |
| HTTP 客户端 | `fetch` 或 axios | 文件上传用 axios 更方便 |
| Markdown 渲染 | `markdown-it` + `highlight.js` | AI 回答通常是 Markdown 格式 |
| 路由 | Vue Router 4 | 聊天页 + 知识库管理页 |
| 状态管理 | Pinia 或 composables | 简单场景用 composables 即可 |

### 2.2 项目结构

```
frontend/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── src/
│   ├── main.ts
│   ├── App.vue
│   ├── router/
│   │   └── index.ts              # /chat 和 /knowledge 两个路由
│   ├── views/
│   │   ├── ChatView.vue          # 对话主界面
│   │   └── KnowledgeView.vue     # 知识库管理
│   ├── components/
│   │   ├── chat/
│   │   │   ├── ChatMessage.vue   # 单条消息气泡（Markdown 渲染）
│   │   │   ├── ChatInput.vue     # 输入框 + 发送按钮
│   │   │   ├── SourceCard.vue    # 引用来源展示卡片
│   │   │   └── SessionList.vue   # 左侧会话列表
│   │   └── knowledge/
│   │       ├── DocumentList.vue  # 文档列表
│   │       └── UploadDialog.vue  # 上传弹窗（拖拽+进度条）
│   ├── composables/
│   │   ├── useWebSocket.ts       # WebSocket 连接 + 消息收发
│   │   └── useKnowledge.ts       # 知识库 API 调用
│   └── styles/
│       └── main.css              # 全局样式
```

### 2.3 关键组件设计

#### A. `composables/useWebSocket.ts` — WebSocket 封装

```typescript
// 核心逻辑：
// - new WebSocket(`ws://localhost:8000/ws/chat?session_id=xxx`)
// - onmessage 解析 JSON，更新响应式状态
// - send(content) 发送用户消息
// - 自动重连机制
// - 暴露: messages, isConnecting, isStreaming, send()

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: { index: number; text: string }[]
  timestamp: number
}
```

#### B. `ChatView.vue` — 对话主界面

布局：左侧会话列表（窄）+ 右侧对话区（宽）

```
┌──────────┬─────────────────────────────────┐
│ 会话列表  │  消息列表                         │
│  + 新会话 │  ┌─────────────────────────┐    │
│  会话1    │  │ 用户: LangChain是什么？   │    │
│  会话2    │  │ AI: LangChain是...       │    │
│  会话3    │  │ [来源1] [来源2] [来源3]   │    │
│           │  └─────────────────────────┘    │
│           │  ┌─────────────────────────┐    │
│           │  │ [输入框]          [发送]  │    │
│           │  └─────────────────────────┘    │
└──────────┴─────────────────────────────────┘
```

**状态展示**：在 AI 回答到达前，显示当前阶段标签（"正在检索..." → "正在生成..."），让用户知道系统在工作。

#### C. `ChatMessage.vue` — 消息气泡

- 用户消息：右对齐，纯文本
- AI 消息：左对齐，Markdown 渲染（代码高亮、表格、列表）
- 来源引用：底部小卡片，点击可展开原文
- 头像占位：用户图标 vs AI 机器人图标

#### D. `KnowledgeView.vue` — 知识库管理

```
┌────────────────────────────────────────────┐
│  知识库管理                                  │
│  ┌──────────────────────────────────────┐  │
│  │  📊 6 篇文档 · 2564 个片段           │  │
│  └──────────────────────────────────────┘  │
│                                             │
│  [上传文档] 按钮                            │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │ 📄 NLP文本预处理.md        46KB   🗑️  │  │
│  │ 📄 RNN及其变体.md           28KB   🗑️  │  │
│  │ 📄 RAG技术-详细版.md       210KB   🗑️  │  │
│  │ ...                                  │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

**上传弹窗**支持拖拽上传 + 进度条 + 多文件批量上传。

### 2.4 路由设计

```typescript
// router/index.ts
const routes = [
  { path: '/', redirect: '/chat' },
  { path: '/chat', component: ChatView },
  { path: '/chat/:sessionId', component: ChatView },  // 恢复特定会话
  { path: '/knowledge', component: KnowledgeView },
]
```

---

## 3. 关键风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| **SentenceTransformer 加载阻塞** | 启动需 10-30s，第一次请求超时 | startup 事件中预加载，加健康检查端点 |
| **ChromaDB SQLite 单写锁** | 多人同时上传报错 | `asyncio.Lock` 串行化写操作 |
| **CPU 密集型任务阻塞事件循环** | embedding/BM25/Cross-Encoder 阻塞其他请求 | `asyncio.to_thread` 或 `run_in_executor` |
| **BM25 索引需重建** | 上传新文档后检索不到新内容 | 上传后调用 `BM25Retriever.from_texts(all_chunks)` 重建 |
| **DeepSeek 流式兼容性** | `astream_events` 可能不支持 | 先用方案 A（伪流式），验证后再升级 |
| **大文件上传超时** | 61MB PDF 分块+向量化可能超 30s | 前端显示上传进度，后端返回任务 ID 异步处理 |

---

## 4. 实施步骤（建议顺序）

### 第 1 步：FastAPI 骨架（~2h）
- 创建 `backend/` 目录结构
- 编写 `main.py`（FastAPI + CORS + startup/shutdown）
- 调整 `vector_store.py` 为函数式初始化
- 编写 `POST /api/chat`（非流式，先跑通同步版）
- 用 `curl` 或 Swagger UI 验证

### 第 2 步：WebSocket 流式（~2h）
- 实现 `/ws/chat` WebSocket 端点
- 对接 LangGraph `app.astream_events()`（或 `asyncio.to_thread` + `app.invoke()`）
- 定义消息协议

### 第 3 步：知识库 API（~1.5h）
- `GET /api/knowledge/documents`
- `POST /api/knowledge/upload`
- `DELETE /api/knowledge/{doc_id}`
- `GET /api/knowledge/stats`

### 第 4 步：Vue 项目初始化（~1h）
- `npm create vite@latest frontend -- --template vue-ts`
- 安装依赖：`vue-router`, `markdown-it`, `highlight.js`
- 配置 Vite 代理（`/api` 和 `/ws` 转发到 FastAPI）

### 第 5 步：聊天界面（~3h）
- `useWebSocket.ts` composable
- `ChatView.vue` 布局
- `ChatMessage.vue`（Markdown 渲染）
- `ChatInput.vue`（输入发送）
- `SessionList.vue`（会话列表）
- 端到端联调

### 第 6 步：知识库管理界面（~2h）
- `useKnowledge.ts` composable（axios 调用后端 API）
- `KnowledgeView.vue` 布局
- `DocumentList.vue`（列表+删除）
- `UploadDialog.vue`（上传+进度条）

### 第 7 步：打磨（~2h）
- SqliteSaver 替换 MemorySaver
- 错误处理 + 加载状态
- 样式美化
- Docker Compose 部署配置（可选）

---

## 5. 验证方法

1. **后端单元验证**：启动 FastAPI → 打开 `http://localhost:8000/docs` → 用 Swagger UI 测试聊天和上传接口
2. **WebSocket 验证**：用浏览器控制台或 Postman WebSocket 客户端连接 `ws://localhost:8000/ws/chat?session_id=test`
3. **前端联调**：`npm run dev` 启动前端 → 在聊天页输入问题 → 观察 WebSocket 消息收发 + Markdown 渲染
4. **知识库流程**：上传新 PDF → 确认列表中出现 → 在聊天中测试新文档是否被检索到
5. **多会话验证**：开两个浏览器标签页 → 不同 session_id → 确认对话不串扰
