# 基于可插拔策略架构的 RAG 智能问答系统

## 1. 项目概述

**名称**：基于可插拔策略架构的 RAG 智能问答系统（FastAPI + Vue 3 Web版）

**形态**：前后端分离的 Web 应用 —— FastAPI 提供 REST / WebSocket 后端，Vue 3 + Ant Design 提供Web交互页面。

**核心能力**：

- 意图路由（闲聊 vs 知识查询）
- 问题复杂度判断 + 自动分解子问题
- RRF 多路召回融合（向量 + BM25）+ 语义相似度去重
- Cross-Encoder 精排
- 反思自纠错重试机制（≤2 次）
- 多轮对话记忆持久化（MemorySaver + 会话级 thread_id 沙箱隔离）
- WebSocket 实时问答、阶段状态推送、流式中断
- 知识库 Web 化管理：上传 / 删除 / 统计，支持 PDF、Word、Markdown、TXT
- 引用溯源：正文 `[^N]` 上角标 + 文末参考来源清单
- 可插拔知识库：代码与领域零耦合，替换文档即切换知识库领域

---

## 2. 系统架构（总体）

```
┌────────────────────────────────────────────────────────────────┐
│                          Vue 3 前端 (5173)                       │
│                                                                  │
│  视图层     ChatView（对话） · KnowledgeView（知识库）             │
│  组合式函数  useWebSocket · useKnowledge · useTheme              │
│  组件层     SessionList · ChatInput · ChatMessage · SourceCard   │
│             DocumentList · UploadDialog                          │
│  样式层     tokens.css（CSS 变量 + 明暗双主题） · glass.css        │
│  路由       /chat · /chat/:sessionId · /knowledge                │
└───────────────┬────────────────────┬────────────────────────────┘
                │ WS /ws/chat         │ REST /api/knowledge
                │                     │
┌───────────────▼────────────────────▼────────────────────────────┐
│                     FastAPI 后端 (8000)                          │
│                                                                  │
│  API 层      api/chat.py（WebSocket） · api/knowledge.py（REST） │
│  ──────────────────────────────────────────────────────────────  │
│  编排层      core/agent_graph.py（LangGraph StateGraph）         │
│              router → agent / chatbot → reflection（条件重试）    │
│  ──────────────────────────────────────────────────────────────  │
│  业务层      core/rag_engine.py（RRF 融合 · 语义去重 · 分解 · 路由）│
│  ──────────────────────────────────────────────────────────────  │
│  基础设施层   vector_store.py（ChromaDB + 双检索器）               │
│              document_loader.py（PDF/MD/TXT/DOCX 解析 · 分块）    │
│  ──────────────────────────────────────────────────────────────  │
│  配置层      core/config.py（LLM · Embedding · CrossEncoder）     │
│              core/__init__.py（延迟初始化单例）                    │
└──────────────────────────────────────────────────────────────────┘
```

**进程模型**：前后端各自独立进程，Vite 开发服务器通过 proxy 把 `/ws`（含 WS 升级）与 `/api` 转发到 FastAPI 端口，前端不直接跨域。

---

## 3. 后端架构分层

### 3.1 API 层（backend/api/）

| 文件 | 职责 | 端点 |
|---|---|---|
| `chat.py` | WebSocket 实时问答 | `WS /ws/chat?session_id=xxx` |
| `knowledge.py` | 知识库 CRUD | `GET/POST/DELETE /api/knowledge/*` |

**WebSocket 消息协议**：

```
客户端 → 服务端: {"type": "query", "content": "用户问题"}
服务端 → 客户端:
  {"type": "status", "stage": "routing|generating"}        # 阶段状态（可被中断）
  {"type": "answer", "content": "完整回答", "sources": [...]} # 最终答案 + 引用列表
  {"type": "error", "message": "错误信息"}
```

**REST API（知识库）**：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/knowledge/documents` | 列出已入库文档（名称/大小/修改时间） |
| POST | `/api/knowledge/upload` | 上传文档，触发分块 + 向量化 + BM25 重建 |
| DELETE | `/api/knowledge/{filename}` | 删除文档及对应向量（按 metadata source 定位） |
| GET | `/api/knowledge/stats` | 文档总数 + 知识片段总数 |

### 3.2 编排层（core/agent_graph.py）

LangGraph StateGraph 节点：

- **router**：三层混合意图仲裁（关键词 + 语义相似度 + LLM 兜底），新请求进入前清空上一轮残留的 `search_hint`
- **agent_node**：LangChain Agent 自主决策工具调用（分解 / 搜索 / 合成）；异常时降级为「直接检索 + 快速生成」
- **chatbot_node**：闲聊节点，直接 LLM 对话
- **reflection_node**：post-hoc 质量评估，输出 `需要重试|优化检索关键词：xxx` 或 `不需要重试`
- **should_retry**：条件边，最多重试 2 次

```
START → router ─┬─ agent → reflection ─┬─ 合格 → END
                │                      └─ 不合格(≤2次) → agent
                └─ chatbot → END
```

图使用 `MemorySaver` 作为 checkpointer，`thread_id` 即会话 ID，实现会话级上下文沙箱隔离。

### 3.3 业务层（core/rag_engine.py）

- `parallel_vec_BM25_retriever`：ThreadPoolExecutor(2) 并行召回向量 + BM25
- `reciprocal_rank_fusion`：RRF(k=60) 融合双路排序
- `deduplicate_semantic`：RRF 融合后按 embedding 余弦相似度去重（阈值 0.85），剔除语义冗余片段
- `my_rag_retrieve_with_meta`：返回 `[{text, source, type}]`，供 Agent 引用溯源
- `semantic_intent` / `llm_router`：语义二分类 + LLM 兜底意图仲裁
- `decompose_query`：LLM 判断复杂度并拆分为 ≤4 个子问题

### 3.4 基础设施层

**vector_store.py**
- ChromaDB `PersistentClient`（路径 `./chroma_db_agent`），空库时才批量编码入库
- 向量检索器 `k=10`，BM25 检索器 `k=20`
- `add_document`：增量单文档入库（分块 → 编码 → 批量写入）
- `rebuild_bm25`：上传/删除后重建 BM25 索引并同步单例

**document_loader.py**

| 格式 | 解析方式 | 类型标记 |
|---|---|---|
| PDF | PyMuPDF (`fitz`) 逐页抽取文本 | `PDF` |
| Markdown / TXT | 直接读文本 + Markdown 清洗 | `Markdown` / `TXT` |
| DOCX | python-docx 提取段落纯文本 | `Word` |

多文件 `ThreadPoolExecutor(max_workers=4)` 并行加载，去重后统一用 `RecursiveCharacterTextSplitter(chunk_size=500, overlap=100)` 分块。

### 3.5 配置层（core/config.py + core/__init__.py）

- **延迟初始化单例**：所有重量级资源（向量库、检索器、LangGraph app）由 `core.init()` 统一初始化，由 FastAPI lifespan startup 调用，避免 `import` 即加载模型
- LLM：DeepSeek（`deepseek-v4-flash`，base_url `api.deepseek.com`），temperature 0.2
- Embedding：`shibing624/text2vec-base-chinese`（768 维）
- CrossEncoder：`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- CPU 线程配置：8 物理核，HF 镜像 `hf-mirror.com`

---

## 4. 前端架构

### 4.1 技术栈

| 层 | 选型 |
|---|---|
| 框架 | Vue 3（Composition API + `<script setup>`）+ TypeScript |
| 构建 | Vite（dev proxy → 8000） |
| UI | Ant Design Vue + @ant-design/icons-vue（矢量图标） |
| 渲染 | markdown-it + highlight.js（Markdown 渲染 + 代码高亮） |
| 路由 | vue-router（history 模式） |
| 状态 | 本地组合式函数 + localStorage 持久化 |

### 4.2 设计系统（黑白灰毛玻璃风）

- **主题**：明暗双主题，全部走 `tokens.css` 的 CSS 变量，`[data-theme="dark"]` 覆盖
- **配色语义**：`--bg-page`（页面）、`--bg-sidebar`（侧栏）、`--text-primary/secondary/tertiary`（文字层级）、`--border-*`（边框层级）
- **毛玻璃**：仅保留输入框/弹窗等关键交互元素使用 `backdrop-filter: blur(12px)`（`.glass-input`），其余区域用纯色背景，克制不花哨
- **字体**：Space Grotesk（标题）/ Open Sans（正文）/ JetBrains Mono（代码、数字）
- **排版**：对话区 `max-width: 1000px`，正文 16px、行高 1.8，回答底部与输入框留 48px 间距

### 4.3 核心交互

| 功能 | 实现 |
|---|---|
| 多会话 | 左侧会话列表，支持新建/置顶/重命名/删除，localStorage 持久化 |
| 会话隔离 | 切换会话 → `useWebSocket` 断开旧连接 → 以新 `session_id` 重连 |
| 问答 | WS 推送 status → 伪流式打字机效果（30ms/字） |
| 中断 | 「停止」按钮 → 清状态消息 + 关闭 WS + 立即重连 |
| 引用溯源 | 正文 `[^N]` 上角标可点击，文末来源卡片可展开原文片段，前端按 index 去重 |
| 追问 | 对当前回答引用式追问，输入框固定显示 |
| 操作 | 复制 / 重新生成 / 追问，最新一条常显，历史消息 hover 显示 |
| 知识库 | 文档列表 + 统计卡片（文档总数/知识片段）+ 上传弹窗（格式校验） |

---

## 5. 检索 Pipeline

```
Query
  │
  ├─► Router（意图分流）
  │     keyword + cosine + LLM 三层仲裁 → 闲聊 / 知识查询
  │
  ├─► Agent Node（LLM Agent 自主决策）
  │     ┌──────────────────────────────────────────────┐
  │     │ 工具1: search_knowledge_tool(query)          │
  │     │   → vector(k=10) ∥ BM25(k=20)（并行）        │
  │     │   → RRF(k=60) → 语义去重(0.85) → CrossEncoder │
  │     │   → 阈值 >0.25 取 top 5，带文档名/类型        │
  │     │                                              │
  │     │ 工具2: decompose_question_tool(query)        │
  │     │   → LLM 判断复杂度 → ≤4 个子问题              │
  │     └──────────────────────────────────────────────┘
  │     Agent 自主决定: 先分解 or 直接搜索？搜几次？如何综合？
  │
  └─► Reflection（反思评估）
        ├─ 合格 → 输出（引用 [^N] + 文末参考来源）
        └─ 不合格 → search_hint → agent 重试 (max 2)

异常降级链路:
  Agent失败 → 跳过Agent, 直接检索+生成 → 仍失败 → 友好提示
```

**关键参数**：

| 参数 | 值 |
|---|---|
| RRF k | 60 |
| RRF + 语义去重阈值 | 0.85 |
| Cross-Encoder 输出 | 置信度 > 0.25 取 top 5 |
| 反思重试上限 | 2 次 |
| 子问题上限 | 4 个 |
| 向量检索 k | 10 |
| BM25 检索 k | 20 |
| 分块大小 / 重叠 | 500 / 100 |
| 并行召回 | ThreadPoolExecutor, max_workers=2 |
| 并行文档加载 | ThreadPoolExecutor, max_workers=4 |
| Agent 工具 | search_knowledge + decompose_question |

---

## 6. 数据流（Web 全链路示例）

```
用户提问（前端）── WS /ws/chat?session_id=abc ──► FastAPI chat.py
  │  线程池隔离: config.thread_id = "abc"（会话沙箱）
  ▼
core.langgraph_app.invoke（asyncio.to_thread 防阻塞）
  ├─ router ──► 关键词"区别"+"是什么" → agent_node
  ├─ agent_node ──► 调用 decompose_question_tool → 3 个子问题
  │                   └─► 逐子问题调用 search_knowledge_tool
  │                        └─► vector ∥ BM25 → RRF → 去重 → 重排 → top 5
  │                            每条带 {source, type}
  ├─ reflection ──► 合格
  └─ 返回 answer（含 [^N] 上角标 + 参考来源清单）
      │
      ▼
chat.py 提取 sources → {"type":"answer", content, sources} ──► 前端
  ▼
前端: 打字机渲染 answer，[^N] 渲染为上角标，文末渲染来源卡片
```

---

## 7. 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 后端框架 | FastAPI + lifespan 延迟初始化 | 重量级模型避免 import 即加载，生命周期显式管理 |
| 前后端通信 | WebSocket 问答 + REST 管理 | 问答需实时推送状态、可中断；知识库 CRUD 用 REST 更直观 |
| 会话隔离 | LangGraph `thread_id` = 会话 ID | MemorySaver 天然按 thread_id 隔离上下文，无需自建会话存储 |
| 中断实现 | WS 关闭 + 立即重连（伪流式） | DeepSeek 流式改造前用打字机模拟；`asyncio.to_thread` 防阻塞事件循环 |
| 检索融合 | RRF（非加权求和）| 对排序位置敏感，不要求分数归一化，对异构检索器更鲁棒 |
| 语义去重 | RRF 后 embedding 相似度 >0.85 剔除 | 消除同一文档多片段语义冗余，提高送入重排的多样性 |
| 问题分解 | LLM 自动判断 | 避免硬编码规则，LLM 理解语义复杂度 |
| Agent 模式 | LangChain Agent 工具自主调用 | LLM 自主决定搜索次数/顺序，比 if-else 灵活 |
| 容错降级 | 三层降级兜底 | Agent 失败 → 直接检索+生成；检索失败 → 友好提示 |
| 增量入库 | `add_document` + `rebuild_bm25` | 上传/删除不触发全库重建，O(单文档) 开销 |
| 按源删除 | metadata `source` 定位 chunk | 避免全量重建带来的状态损坏风险 |
| 知识库格式 | PDF / Markdown / TXT / Word（纯文本） | 覆盖常见办公/技术文档；DOCX 先取段落纯文本 |
| 前端状态 | localStorage 持久化会话/历史 | 刷新、路由切换不丢数据，无需服务端会话 |
| 引用溯源 | `[^N]` 上角标 + 文末来源清单 | 回答可验证、可追溯，符合 RAG 最佳实践 |

---

## 8. 演进历程

| 阶段 | V1（CLI 加权融合） | V2（RRF+分解） | V3（Agent 模式） | V4（并行化） | **V5（Web 化）** |
|---|---|---|---|---|---|
| 入口 | CLI 对话循环 | CLI | CLI | CLI | **FastAPI + Uvicorn** |
| 前端 | 无 | 无 | 无 | 无 | **Vue 3 + Ant Design** |
| 召回 | EnsembleRetriever | vector + BM25 独立 | vector + BM25 独立 | ThreadPoolExecutor 并行 | 并行 + **语义去重(0.85)** |
| 融合 | 权重 [0.5, 0.5] | RRF (k=60) | RRF (k=60) | RRF (k=60) | RRF (k=60) |
| 决策者 | 代码 | 代码 | LLM Agent | LLM Agent | LLM Agent |
| 记忆 | 无 | 无 | MemorySaver | MemorySaver | **thread_id 会话沙箱隔离** |
| 知识库管理 | 手动放文件 | 手动放文件 | 手动放文件 | 手动放文件 | **Web 上传/删除/统计** |
| 格式支持 | md/pdf | md/pdf | md/pdf | md/pdf | **md/txt/pdf/docx** |
| 引用溯源 | 无 | 无 | 无 | 简单溯源 | **[^N] 上角标 + 来源清单** |

---

## 9. 文件清单

```
Alter-Docs-Rag/
├── backend/                     # FastAPI 后端
│   ├── main.py                  # 入口：lifespan 初始化 + CORS + 路由注册
│   ├── requirements.txt
│   ├── api/
│   │   ├── chat.py              # WebSocket 聊天端点 + 来源解析
│   │   └── knowledge.py         # 知识库 REST（文档/上传/删除/统计）
│   ├── core/
│   │   ├── __init__.py          # 延迟初始化单例（core.init）
│   │   ├── config.py            # LLM / Embedding / CrossEncoder / 分块器
│   │   ├── agent_graph.py       # LangGraph 图：router/agent/chatbot/reflection
│   │   ├── rag_engine.py        # RRF 融合 · 语义去重 · 意图路由 · 分解
│   │   ├── vector_store.py      # ChromaDB · 双检索器 · 增量入库 · BM25 重建
│   │   └── document_loader.py   # PDF/MD/TXT/DOCX 解析 · 清洗 · 并行分块
│   ├── models/schemas.py        # Pydantic 模型
│   ├── knowledge_base/          # 知识库源文档（md/pdf/docx）
│   ├── chroma_db_agent/         # 持久化向量库
│   └── .env                     # API 密钥
└── frontend/                    # Vue 3 前端
    ├── vite.config.ts           # dev proxy: /ws + /api → 8000
    ├── src/
    │   ├── App.vue              # 根组件（路由出口）
    │   ├── main.ts
    │   ├── router/index.ts      # /chat · /chat/:sessionId · /knowledge
    │   ├── views/
    │   │   ├── ChatView.vue     # 对话主视图（会话管理 + 消息流）
    │   │   └── KnowledgeView.vue# 知识库视图（文档列表 + 统计）
    │   ├── components/
    │   │   ├── chat/            # SessionList · ChatInput · ChatMessage · SourceCard
    │   │   └── knowledge/       # DocumentList · UploadDialog
    │   ├── composables/         # useWebSocket · useKnowledge · useTheme
    │   └── styles/              # tokens.css（双主题变量）· glass.css（毛玻璃）
```
