# 面向AI开发技术文档可插拔知识库的 LangGraph 自主决策式自省 RAG Agent

---

## 核心亮点 / Highlights

| 维度 | 做了什么 | 为什么值得写 |
|---|---|---|
| **检索精度** | 从加权融合升级为 RRF 倒数排名融合 + Cross-Encoder 精排 | 独立理解向量检索与稀疏检索的互补性，通过非归一化的融合策略避免分数尺度不一致问题 |
| **Agent 自主决策** | 给 LLM 两个工具（`search_knowledge` + `decompose_question`），Agent 自主决定先分解还是直接搜索、搜几次、如何综合 | 展示了真正的 Agent 模式——LLM 自主规划工具调用序列，而非硬编码 if-else 流程 |
| **问题分解** | LLM 自动判断查询复杂度，复杂问题拆分为 2-4 个子问题分别检索 | 将单步检索泛化为多步检索 pipeline，Agent 可根据内容灵活调整检索策略 |
| **反思自纠错** | 生成后评估回答质量，不合格时自动优化检索词并重试（最多 2 次）| 引入 Self-Reflection 闭环，在不增加外部监督的情况下提升答案可靠性 |
| **代码工程化** | 从 650 行单文件重构为 6 模块分层架构（配置层→基础设施层→业务层→编排层→入口层） | 展示了模块解耦、单向依赖、接口设计的能力 |
| **容错降级** | Agent 调用失败自动降级为直接检索+生成，检索失败返回友好提示——三层兜底，单点故障不崩链路 | 体现了生产级系统的健壮性设计思维 |
| **可插拔知识库** | 代码与领域零耦合，`knowledge_base/` 目录热插拔，替换文档即切换至任意领域 | 系统设计不绑定特定领域，体现通用工具思维而非一次性作业 |

## 系统指标 / Metrics

| 指标 | 数值 |
|---|---|
| 知识库文档 | 当前 6 篇（MD + PDF），当前覆盖 AI 开发技术文档（可替换为任意领域） |
| 文本片段 | 当前 2,564 chunks（chunk_size=500, overlap=100） |
| 检索延迟 | 向量 ∥ BM25 双路并行召回 < 1s（ThreadPoolExecutor），Cross-Encoder 重排 < 200ms |
| 文档加载 | 多文件并行IO加载（ThreadPoolExecutor, max_workers=4），6 篇文档 < 5s |
| 重试机制 | 最多 2 次反思重试，每次生成优化检索词 |
| 代码规模 | 6 模块、~400 行核心逻辑、零循环依赖 |

## 检索 Pipeline

```
Query
  │
  ├─► Router（意图分流）
  │     keyword + cosine + LLM 三层仲裁 → 闲聊 / 知识查询
  │
  ├─► Agent Node（LLM Agent 自主决策）
  │     ┌─────────────────────────────────────┐
  │     │ 工具1: search_knowledge(query)       │
  │     │   → vector(15) ∥ BM25(15)（并行）            │
  │     │   → RRF(k=60) → Cross-Encoder → top 5│
  │     │                                      │
  │     │ 工具2: decompose_question(query)     │
  │     │   → LLM 判断复杂度 → 子问题列表       │
  │     └─────────────────────────────────────┘
  │     Agent 自主决定: 先分解 or 直接搜索？搜几次？如何综合？
  │
  └─► Reflection（反思评估）
        ├─ 合格 → 输出
        └─ 不合格 → search_hint → 重试 (max 2)

异常降级链路:
  Agent失败 → 跳过Agent, 直接检索+生成 → 仍失败 → 友好提示
```

## 技术栈 / Tech Stack

```
┌─────────────────────────────────────────────┐
│ LangGraph (StateGraph + MemorySaver)         │  编排 & 状态持久化
├─────────────────────────────────────────────┤
│ DeepSeek V4 Flash                            │  LLM 生成 + 路由 + 分解 + 反思
│ text2vec-base-chinese                        │  Embedding（768维）
│ mmarco-mMiniLMv2-L12-H384-v1                 │  Cross-Encoder 重排
├─────────────────────────────────────────────┤
│ ChromaDB (PersistentClient)                  │  向量存储 & 检索
│ BM25 (langchain_community)                   │  稀疏关键词检索
│ RRF (Reciprocal Rank Fusion)                 │  多路融合，k=60
├─────────────────────────────────────────────┤
│ PyMuPDF · RecursiveCharacterTextSplitter     │  文档解析 & 中文分块
│ SentenceTransformer · Chroma · LangChain     │  基础框架
└─────────────────────────────────────────────┘
```

## 关键设计决策 / Why This Way

| 决策点 | 我的选择 | 如果不这样做会怎样 |
|---|---|---|
| 检索融合用 **RRF** 而非加权求和 | RRF 不需要分数归一化，向量相似度与 BM25 的分数尺度完全不同，加权求和会偏袒某一方 | EnsembleRetriever 的 [0.5, 0.5] 权重实际上是粗暴的平均，没有考虑两个检索器返回的分数分布差异 |
| LLM 判断复杂度而非规则 | 规则只能靠关键词，遇到"对比一下 RNN 和 Transformer"这种不带明确标志词的比较句会漏判 | 关键词规则覆盖不全，复杂查询被当作简单查询处理，召回缺失一半信息 |
| LangChain Agent 工具自主调用 | Agent 自主决定先分解还是直接搜索、搜几次、如何综合——比硬编码 if-else 更灵活，且能处理规则覆盖不到的边界情况 | 硬编码编排遇到"对比一下 RNN 和 Transformer"这种不带明确标志词的比较句时，不会触发分解，召回缺失一半信息 |
| ChromaDB 持久化而非内存 | 启动时检查 `collection.count() == 0`，仅空库时嵌入写入，后续直接加载 | 内存模式每次重启都要重新 embedding（~2564 chunks 约 30 秒），且数据不持久 |
| Post-hoc 反思而非复杂 self-correct | 评估 → 失败 → 优化关键词 → 重试，简单明确，token 开销可控 | Self-RAG / CRAG 需要多次 LLM 调用评估每个 chunk 的相关性，token 消耗巨大，且容易过度修正 |

## 架构 / Architecture

```
入口层:     main.py            对话循环
             │
编排层:     agent_graph.py      LangGraph StateGraph
             ├── router         三层仲裁分流
             ├── agent_node     LLM Agent 自主决策工具调用
             ├── chatbot_node   闲聊对话
             └── reflection     反思评估 + 条件重试
             │
业务层:     rag_engine.py       RRF · my_rag_retrieve · decompose_query
             │
基础设施:   vector_store.py     ChromaDB · Vector Retriever ∥ BM25 Retriever（并行）
             document_loader.py  PDF/MD 解析 · Markdown清洗 · 中文分块（多文件并行IO）
             │
配置层:     config.py           LLM · Embedding · Cross-Encoder · 环境变量
```

> 详细设计文档：[DESIGN.md](./DESIGN.md) — 包含完整数据流、LangGraph 状态图、参数规格

## 演示 / Demo

```
用户: LangChain和LangGraph是什么，有什么区别？

[Router] 关键词命中 "区别"+"是什么" → 路由到知识查询
[Agent] LLM 分析: 涉及对比 → 调用 decompose_question
        → 返回 3 个子问题

[Agent] 调用 search_knowledge("LangChain技术原理与适用场景")
        → vector∥BM25（并行）→ RRF → Cross-Encoder → top 5  ✓
[Agent] 调用 search_knowledge("LangGraph技术原理与适用场景")
        → vector∥BM25（并行）→ RRF → Cross-Encoder → top 5  ✓
[Agent] 调用 search_knowledge("LangChain与LangGraph核心区别")
        → vector∥BM25（并行）→ RRF → Cross-Encoder → top 5  ✓

[Agent] 综合 15 chunks → 生成结构化对比回答

[Reflection] 评估: 回答合格 ✓

AI: LangChain 和 LangGraph 的核心区别在于...
    [原理对比 + 适用场景 + 联系总结 + 来源标注]
```

## 项目结构

```
├── main.py                # 入口：对话循环
├── agent_graph.py          # 编排层：LangGraph 节点 + 图组装
├── rag_engine.py           # 业务层：RRF 融合、检索、问题分解、意图路由
├── vector_store.py         # 基础设施：ChromaDB、vector/BM25 检索器
├── document_loader.py      # 基础设施：文档加载、清洗、分块
├── config.py               # 配置层：LLM、模型、环境变量
├── DESIGN.md               # 系统设计文档
├── knowledge_base/         # 知识库源文档（6篇）
├── chroma_db_agent/        # 持久化向量库
└── .gitignore
```

## 快速开始

```bash
# 1. 安装依赖
pip install langgraph langchain langchain-openai langchain-community \
            langchain-classic langchain-chroma langchain-text-splitters \
            chromadb sentence-transformers pymupdf python-dotenv

# 2. 配置 API Key (.env)
echo 'api_key = "your-api-key"' > .env

# 3. 放入知识库文档 (knowledge_base/)
# 支持 .md / .txt / .pdf

# 4. 启动
python main.py
```

首次运行自动完成文档加载→清洗→分块→向量化→写入 ChromaDB，后续启动直接复用持久化数据。
