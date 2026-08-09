# RAG 智能问答系统 — FastAPI + Vue 3 Web 版

> 基于可插拔策略架构的 RAG 智能问答系统。LangGraph 编排 + ChromaDB 向量检索 + RRF 多路融合，带黑白灰毛玻璃科技风的 Web 交互界面。

![architecture](https://img.shields.io/badge/backend-FastAPI-009688) ![frontend](https://img.shields.io/badge/frontend-Vue3%20%2B%20TS-42b883) ![rag](https://img.shields.io/badge/RAG-LangGraph%20%2B%20ChromaDB-4f46e5)

---

## 核心亮点 / Highlights

| 维度 | 做了什么 | 为什么值得写 |
|---|---|---|
| **检索精度** | RRF 倒数排名融合 + 语义相似度去重（0.85）+ Cross-Encoder 精排 | 向量与稀疏检索互补，去重消除语义冗余，重排提升 top-5 质量 |
| **Agent 自主决策** | 给 LLM 两个工具（`search_knowledge_tool` + `decompose_question_tool`），自主决定分解/搜索/综合 | 真正的 Agent 模式——LLM 自主规划工具调用序列，而非硬编码 if-else |
| **会话级沙箱隔离** | WebSocket 携带 `session_id` → LangGraph `thread_id`，MemorySaver 天然隔离会话上下文 | 多会话并行不串扰，切换会话即切换独立上下文容器 |
| **Web 化知识库** | FastAPI 提供上传/删除/统计接口，前端可视化管理文档 | 从手动放文件升级为浏览器一键入库，格式支持 PDF/Word/MD/TXT |
| **引用溯源** | 正文 `[^N]` 上角标标注 + 文末完整参考来源清单（文档名/类型/原文片段） | RAG 回答可验证、可追溯，提升可信度 |
| **流式中断** | 打字机效果模拟流式 + 「停止」按钮实时中断，保留已生成内容 | 长回答可随时中断，交互不卡死 |
| **双主题设计系统** | CSS 变量驱动明暗主题，黑白灰为主色，毛玻璃仅用于关键交互元素 | 克制的高级感：白底 + 浅灰侧栏，参照 GPT / DeepSeek 布局 |
| **容错降级** | Agent 失败 → 直接检索+生成 → 检索失败 → 友好提示，三层兜底 | 单点故障不崩整个链路 |
| **可插拔知识库** | 代码与领域零耦合，`knowledge_base/` 目录热插拔 | 替换文档即切换至任意领域，无需修改业务逻辑 |

---

## 功能特性 / Features

### 对话

- 🗂️ **多会话管理**：新建、置顶（📌 固定图标）、重命名、删除；切换会话互不干扰
- ⏱️ **实时状态**：分析中阶段提示 + 打字机效果，随时可「停止」中断
- 📎 **引用溯源**：正文 `[^N]` 上角标，文末来源卡片可展开原文片段
- 🧠 **推理补充**：模型内部知识回答标「推理补充」，与检索内容区分
- 📋 **操作按钮**：复制 / 重新生成 / 追问（引用式追问）
- 🔍 **对话搜索**：按关键词过滤会话历史
- ☀️/🌙 **明暗主题**：一键切换，全部走设计 Token

### 知识库

- 📤 **上传文档**：PDF、Word（.docx）、Markdown、TXT 一键入库
- 🗑️ **删除文档**：按 metadata 精准删除对应向量，实时刷新统计
- 📊 **统计面板**：文档总数、知识片段数实时展示
- 💾 **持久化**：ChromaDB 落盘，重启复用已入库向量

### 检索能力

- 三层意图路由（关键词 + 语义 + LLM 兜底）
- 向量（k=10）∥ BM25（k=20）并行召回 → RRF(k=60) → 语义去重 → 精排 top 5
- 复杂问题自动分解为 ≤4 个子问题分别检索
- 反思评估 + 关键词优化重试（≤2 次）

---

## 系统指标 / Metrics

| 指标 | 数值 |
|---|---|
| 知识库文档 | 支持 PDF / Markdown / TXT / DOCX（当前 7 篇） |
| 文本片段 | chunk_size=500, overlap=100，可增量入库 |
| 检索链路 | 双路并行召回 < 1s + Cross-Encoder 重排 < 200ms |
| 文档加载 | 多文件并行 IO（max_workers=4） |
| 重试机制 | 最多 2 次反思重试 |
| 模型 | DeepSeek（生成/路由/分解/反思）+ text2vec（768 维嵌入）+ mmarco-mMiniLMv2（重排） |

---

## 技术栈 / Tech Stack

```
┌────────────────────────────────────────────────────┐
│ FastAPI + Uvicorn + WebSocket                      │  后端服务 & 实时通信
├────────────────────────────────────────────────────┤
│ LangGraph (StateGraph + MemorySaver)               │  编排 & 会话级状态持久化
│ DeepSeek · text2vec-base-chinese · CrossEncoder    │  LLM / 嵌入 / 重排
├────────────────────────────────────────────────────┤
│ ChromaDB (PersistentClient) · BM25 · RRF           │  向量存储 & 多路融合检索
│ PyMuPDF · python-docx · RecursiveCharacterTextSplitter │ 文档解析 & 中文分块
├────────────────────────────────────────────────────┤
│ Vue 3 + TypeScript + Vite + Ant Design Vue         │  前端框架 & UI
│ markdown-it + highlight.js · vue-router            │  渲染 / 路由
└────────────────────────────────────────────────────┘
```

---

## 快速开始 / Quick Start

### 后端（backend/）

```bash
cd backend

# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
echo 'api_key = "your-deepseek-api-key"' > .env

# 3. 放入知识库文档（可选）
# 放到 backend/knowledge_base/，支持 .md / .txt / .pdf / .docx

# 4. 启动
uvicorn main:app --reload --port 8000
```

> 首次启动自动完成 文档加载 → 清洗 → 分块 → 向量化 → 写入 ChromaDB（耗时约 30s），后续启动直接复用持久化数据。API 文档：<http://localhost:8000/docs>。

### 前端（frontend/）

```bash
cd frontend

# 1. 安装依赖
npm install

# 2. 启动开发服务器
npm run dev
```

访问 <http://localhost:5173>。Vite 已配置 proxy，`/ws`（含 WebSocket）与 `/api` 自动转发到后端 8000 端口。

### 使用流程

1. 打开对话页，提问任意问题 → 路由判断知识查询还是闲聊
2. 需要知识库时：进入「知识库」页上传文档 → 返回对话提问即可检索到
3. 回答中 `[^N]` 上角标对应文末参考来源，点击可查看原文片段

---

## 检索 Pipeline

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

---

## 架构 / Architecture

```
┌──────────────┐        WS /ws/chat          ┌──────────────────────────────┐
│  Vue 3 前端   │ ──────────────────────────▶ │ FastAPI 后端 (8000)           │
│  (5173)      │ ◀────────────────────────── │  api/chat.py (WebSocket)      │
│  会话管理      │   {"type":"answer",...}    │  api/knowledge.py (REST)      │
│  打字机/停止   │        REST /api/knowledge │  ──────────────────────────── │
│  知识库管理    │ ──────────────────────────▶ │  core/agent_graph.py          │
└──────────────┘                             │   LangGraph StateGraph        │
                                              │    router → agent / chatbot  │
                                              │    reflection（条件重试）      │
                                              │  core/rag_engine.py           │
                                              │   RRF · 去重 · 分解 · 路由     │
                                              │  core/vector_store.py         │
                                              │   ChromaDB · vector ∥ BM25    │
                                              │  core/document_loader.py      │
                                              │   PDF/MD/TXT/DOCX · 分块      │
                                              └──────────────────────────────┘
```

> 详细设计文档：[DESIGN.md](./DESIGN.md) — 包含完整数据流、LangGraph 状态图、WebSocket 协议、关键参数规格

---

## 项目结构

```
Alter-Docs-Rag/
├── backend/
│   ├── main.py                  # FastAPI 入口：lifespan 初始化 + CORS + 路由
│   ├── requirements.txt
│   ├── api/
│   │   ├── chat.py              # WebSocket 聊天 + 引用来源解析
│   │   └── knowledge.py         # 知识库 REST：文档/上传/删除/统计
│   ├── core/
│   │   ├── __init__.py          # 延迟初始化单例（core.init）
│   │   ├── config.py            # LLM / Embedding / CrossEncoder / 分块器
│   │   ├── agent_graph.py       # LangGraph：router/agent/chatbot/reflection
│   │   ├── rag_engine.py        # RRF 融合 · 语义去重 · 意图路由 · 分解
│   │   ├── vector_store.py      # ChromaDB · 双检索器 · 增量入库 · BM25 重建
│   │   └── document_loader.py   # PDF/MD/TXT/DOCX 解析 · 清洗 · 并行分块
│   ├── models/schemas.py        # Pydantic 模型
│   ├── knowledge_base/          # 知识库源文档
│   ├── chroma_db_agent/         # 持久化向量库
│   └── .env                     # API 密钥
└── frontend/
    ├── vite.config.ts           # dev proxy: /ws + /api → 8000
    └── src/
        ├── App.vue · main.ts
        ├── router/index.ts      # /chat · /chat/:sessionId · /knowledge
        ├── views/               # ChatView.vue · KnowledgeView.vue
        ├── components/
        │   ├── chat/            # SessionList · ChatInput · ChatMessage · SourceCard
        │   └── knowledge/       # DocumentList · UploadDialog
        ├── composables/         # useWebSocket · useKnowledge · useTheme
        └── styles/              # tokens.css（双主题）· glass.css（毛玻璃）
```

---

## 关键设计决策 / Why This Way

| 决策点 | 我的选择 | 如果不这样做会怎样 |
|---|---|---|
| 检索融合用 **RRF** 而非加权求和 | RRF 不需要分数归一化，向量与 BM25 分数尺度完全不同，加权会偏袒一方 | 粗暴平均权重，忽略两路分数分布差异 |
| RRF 后加**语义去重** | embedding 相似度 >0.85 剔除冗余片段 | 同一文档多片段挤占 top-k，送入精排的信息多样性不足 |
| LangGraph **thread_id = 会话 ID** | MemorySaver 天然按 thread_id 隔离上下文 | 自己维护会话状态，边界容易串扰 |
| **WebSocket + 伪流式** | asyncio.to_thread 跑推理 + 前端打字机 + 可中断 | 长回答无状态反馈，用户只能干等 |
| **延迟初始化单例** | core.init() 由 lifespan startup 调用 | import 即加载模型，拖慢启动、难测试 |
| ChromaDB **持久化**而非内存 | 仅空库写入，后续直接加载 | 每次重启都要重新 embedding（约 30s） |
| 按 metadata **source 删除** | 只删该文档 chunk，不全量重建 | 全量重建失败会污染整个向量库 |
| Post-hoc **反思**而非复杂 self-correct | 评估 → 优化关键词 → 重试，token 可控 | Self-RAG / CRAG 多次评估每个 chunk，token 消耗巨大 |

---

## 路线图 / Roadmap

- [x] FastAPI 后端 + Vue 3 前端全栈化
- [x] WebSocket 实时问答 + 流式中断
- [x] 多会话沙箱隔离 + localStorage 持久化
- [x] 知识库 Web 化管理（上传/删除/统计）
- [x] 引用溯源（`[^N]` 上角标 + 参考来源清单）
- [x] DOCX 纯文本支持
- [ ] DOCX / PDF 表格抽取
