# 面向 AI 开发技术文档的 LangGraph 自省式混合召回 RAG 问答 Agent

---

## 核心亮点 / Highlights

| 维度 | 做了什么 | 为什么值得写 |
|---|---|---|
| **检索精度** | 从加权融合升级为 RRF 倒数排名融合 + Cross-Encoder 精排 | 独立理解向量检索与稀疏检索的互补性，通过非归一化的融合策略避免分数尺度不一致问题 |
| **问题分解** | LLM 自动判断查询复杂度，复杂问题拆分为 2-4 个子问题分别检索后合成 | 将单步检索泛化为多步检索 pipeline，提升对比类/多步推理类问题的召回完整度 |
| **反思自纠错** | 生成后评估回答质量，不合格时自动优化检索词并重试（最多 2 次）| 引入 Self-Reflection 闭环，在不增加外部监督的情况下提升答案可靠性 |
| **代码工程化** | 从 453 行单文件重构为 6 模块分层架构（配置层→基础设施层→业务层→编排层→入口层）| 展示了模块解耦、单向依赖、接口设计的能力 |

## 系统指标 / Metrics

| 指标 | 数值 |
|---|---|
| 知识库文档 | 6 篇（MD + PDF），覆盖 NLP / RAG / Transformer / LangChain / AI Agent |
| 文本片段 | 2,564 chunks（chunk_size=500, overlap=100） |
| 检索延迟 | 向量 + BM25 双路并行召回 < 1s，Cross-Encoder 重排 < 200ms |
| 重试机制 | 最多 2 次反思重试，每次生成优化检索词 |
| 代码规模 | 6 模块、~400 行核心逻辑、零循环依赖 |

## 检索 Pipeline

```
Query
  │
  ├─► Router（意图分流）
  │     keyword 匹配 + 语义相似度 cosine + LLM 兜底仲裁
  │     ──► 闲聊 → Chatbot
  │     ──► 知识查询 ↓
  │
  ├─► decompose_query（复杂度判断）
  │     LLM 分析语义结构，输出 JSON {is_complex, sub_queries[]}
  │     ├─ 简单查询 ──────────────────────────┐
  │     └─ 复杂查询 → 拆 2-4 子问题逐条检索 ──┤
  │                                             │
  ├─► 多路召回 + RRF 融合 ◄────────────────────┘
  │     ├─ vector_retriever (ChromaDB, k=15)
  │     ├─ bm25_retriever  (k=15)
  │     └─ RRF(k=60) → top 15
  │
  ├─► Cross-Encoder 重排
  │     mmarco-mMiniLMv2 逐对打分, score > 0.25 过滤 → top 5
  │
  ├─► LLM 生成 / synthesize 合成
  │     简单：直接基于 top 5 chunks 生成
  │     复杂：子问题 chunks 去重 → 重排 → 结构化合成
  │
  └─► Reflection（反思评估）
        ├─ 合格 → 输出最终答案
        └─ 不合格 → 生成优化检索词 → 重试 (max 2)
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
| 直接编排而非 LangChain Agent | Agent 模式每次需要额外的 LLM 调用来决定 tool 调用，而知识查询场景下确定要检索，这层决策是浪费 | 每次查询多 1 次 LLM 调用（延迟 + token 消耗），且 Agent 可能"忘记"调用工具 |
| ChromaDB 持久化而非内存 | 启动时检查 `collection.count() == 0`，仅空库时嵌入写入，后续直接加载 | 内存模式每次重启都要重新 embedding（~2564 chunks 约 30 秒），且数据不持久 |
| Post-hoc 反思而非复杂 self-correct | 评估 → 失败 → 优化关键词 → 重试，简单明确，token 开销可控 | Self-RAG / CRAG 需要多次 LLM 调用评估每个 chunk 的相关性，token 消耗巨大，且容易过度修正 |

## 架构 / Architecture

```
入口层:     main.py            对话循环
             │
编排层:     agent_graph.py      LangGraph StateGraph
             ├── router         三层仲裁分流
             ├── agent_node     分解 → 检索 → 生成
             ├── chatbot_node   闲聊对话
             └── reflection     反思评估 + 条件重试
             │
业务层:     rag_engine.py       RRF · my_rag_chain · decompose · synthesize
             │
基础设施:   vector_store.py     ChromaDB · Vector Retriever · BM25 Retriever
             document_loader.py  PDF/MD 解析 · Markdown清洗 · 中文分块
             │
配置层:     config.py           LLM · Embedding · Cross-Encoder · 环境变量
```

> 详细设计文档：[DESIGN.md](./DESIGN.md) — 包含完整数据流、LangGraph 状态图、参数规格

## 演示 / Demo

```
用户: LangChain和LangGraph是什么，有什么区别？

[Router] 关键词命中 "区别"+"是什么" → 路由到知识查询
[Decompose] LLM 判断为复杂问题，拆分为 3 个子问题:
  1. LangChain技术原理与适用场景
  2. LangGraph技术原理与适用场景
  3. LangChain与LangGraph的核心区别对比，有什么联系

[Retrieval] 逐子问题检索:
  sub_q1 → vector(15)+BM25(15) → RRF → top 5  ✓
  sub_q2 → vector(15)+BM25(15) → RRF → top 5  ✓
  sub_q3 → vector(15)+BM25(15) → RRF → top 5  ✓

[Synthesize] 15 chunks 去重 → Cross-Encoder 重排(原始问题) → top 5 → LLM 合成

[Reflection] 评估: 回答合格 ✓

AI: LangChain（Agent框架）和LangGraph（Workflow）是两种不同的...
    [结构化回答: 原理对比 + 场景分析 + 联系总结]
```

## 项目结构

```
├── main.py                # 入口：对话循环
├── agent_graph.py          # 编排层：LangGraph 节点 + 图组装
├── rag_engine.py           # 业务层：RRF、RAG链路、问题分解、意图路由
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
