# 基于LangGraph的自省式Rag-Agent 系统设计文档

## 1. 项目概述

**名称**：面向AI开发技术文档可插拔知识库的 LangGraph 自主决策式自省 RAG Agent

**核心能力**：

- 意图路由（闲聊 vs 知识查询）
- 问题复杂度判断 + 自动分解子问题
- RRF 多路召回融合（向量 + BM25）
- Cross-Encoder 精排
- 反思自纠错重试机制
- 多轮对话记忆持久化
- 可插拔知识库：代码与领域零耦合，替换文档即切换知识库领域，无需修改任何业务逻辑

---

## 2. 架构分层

```
┌─────────────────────────────────────────────────────┐
│                    main.py                           │
│                  （入口层）                            │
│              对话循环 + config 注入                    │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 agent_graph.py                       │
│                （编排层 / Controller）                 │
│                                                      │
│   ┌──────────┐  ┌──────────┐  ┌─────────────────┐   │
│   │  router  │  │agent_node│  │ reflection_node │   │
│   │ 意图分流  │─▶│ LLM Agent│─▶│   反思评估       │   │
│   │          │  │ 自主决策  │  │                 │   │
│   └──────────┘  └──────────┘  └─────────────────┘   │
│        │              │               │              │
│   ┌────▼────┐         │          ┌────▼────┐        │
│   │chatbot  │         │          │should_   │        │
│   │闲聊节点  │         │          │retry(≤2) │        │
│   └─────────┘         │          └──────────┘        │
│                       │                              │
│            LangGraph StateGraph                      │
│            MemorySaver (checkpoint)                  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  rag_engine.py                       │
│                （业务逻辑层 / Service）                 │
│                                                      │
│   ┌─────────────────┐  ┌───────────────────┐        │
│   │  意图路由函数      │  │   检索 & 分解       │        │
│   │  semantic_intent │  │  my_rag_retrieve   │        │
│   │  llm_router      │  │  reciprocal_rank_   │        │
│   │  CHAT_KW / KNOW_KW│ │  fusion (RRF)      │        │
│   └─────────────────┘  │  decompose_query   │        │
│                         └───────────────────┘        │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              vector_store.py + document_loader.py     │
│                （基础设施层 / Infrastructure）          │
│                                                      │
│   ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
│   │ ChromaDB   │  │vector_ret- │  │ BM25         │  │
│   │ 持久化向量库 │  │riever(k=15)│  │ retriever    │  │
│   │ (2564条)   │  │            │  │ (k=15)       │  │
│   └────────────┘  └────────────┘  └──────────────┘  │
│                                                      │
│   ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
│   │ load_all_  │  │clean_mark- │  │ load_pdf     │  │
│   │ docs       │  │down        │  │ (PyMuPDF)    │  │
│   │ (并行IO)    │  │            │  │              │  │
│   └────────────┘  └────────────┘  └──────────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                    config.py                          │
│                 （配置层）                              │
│                                                      │
│   llm (DeepSeek V4)  │  embedding_model (text2vec)   │
│   cross_encoder      │  text_splitter (500/100)      │
│   EmbedChunk 适配类   │  CPU 线程配置 (8核)            │
│   HF_ENDPOINT 镜像   │  .env 密钥加载                 │
└─────────────────────────────────────────────────────┘
```

---

## 3. 模块依赖图

```
main.py
  ├── agent_graph.py
  │     ├── config.py (llm, cross_encoder)
  │     └── rag_engine.py
  │           ├── config.py (llm, embedding_model)
  │           └── vector_store.py (vector_retriever, bm25_retriever)
  │                 ├── config.py (embedding_model, EmbedChunk)
  │                 └── document_loader.py (load_all_docs)
  │                       └── config.py (text_splitter)
  └── (无其他依赖)
```

**依赖规则**：严格单向，上层依赖下层，无循环。Python 模块单例缓存保证全局对象只初始化一次。

---

## 4. LangGraph 工作流

```
                    ┌──────────┐
    用户输入 ──────▶│  START   │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │  router  │  三层混合仲裁
                    │ 意图分流  │  keyword + semantic + LLM
                    └┬────────┬┘
                     │        │
             知识查询 │        │ 闲聊
                     │        │
              ┌──────▼──┐  ┌─▼────────┐
              │ agent   │  │ chatbot  │──▶ END
              │ _node   │  │ 直接LLM对话│
              └────┬────┘  └──────────┘
                   │
         ┌─────────▼──────────┐
         │ LangChain Agent     │
         │ 自主决策工具调用      │
         │                     │
         │ 工具1: search_       │
         │   knowledge(query)  │
         │   → RRF + rerank    │
         │   → 返回 top 5      │
         │                     │
         │ 工具2: decompose_    │
         │   question(query)   │
         │   → LLM 判断复杂度   │
         │   → 返回子问题列表   │
         │                     │
         │ Agent 综合 chunks   │
         │ 生成最终答案         │
         └─────────┬──────────┘
                   │
            ┌──────▼──────┐
            │ reflection  │  评估回答质量
            │   _node     │  准确/完整？
            └──┬──────┬───┘
               │      │
        不合格  │      │ 合格
       (≤2次)  │      │
    ┌──────────▼─┐    │
    │  生成       │    │
    │ search_hint │    │
    │ 重试 agent  │    │
    └─────────────┘    │
                       ▼
                      END
```

**关键参数**：

| 参数 | 值 |
|---|---|
| RRF k | 60 |
| RRF + rerank 输出 | top 5 chunks |
| 重排 score 阈值 | > 0.25 |
| 反思重试上限 | 2 次 |
| 子问题上限 | 4 个 |
| 向量检索 k | 15 |
| BM25 检索 k | 15 |
| 分块大小/重叠 | 500 / 100 |
| vec+BM25 并行召回 | ThreadPoolExecutor, max_workers=2 |
| 文档并行加载 | ThreadPoolExecutor, max_workers=4 |
| Agent 工具 | search_knowledge + decompose_question |

---

## 5. 数据流（Agent 工具调用示例）

```
用户: "LangChain和LangGraph有什么区别，各适用于什么场景？"
  │
  ▼
router ──▶ 关键词命中 "区别"+"是什么" → 路由到知识查询
  │
  ▼
agent_node ──▶ invoke LangChain Agent
  │
  ├─► [Agent 决策] 问题涉及对比 → 调用 decompose_question()
  │     └─► LLM 分析: is_complex=True
  │         返回: "1. LangChain技术原理与适用场景
  │                2. LangGraph技术原理与适用场景
  │                3. LangChain与LangGraph的核心区别对比"
  │
  ├─► [Agent 决策] 逐子问题检索 → 调用 search_knowledge(" LangChain技术原理与适用场景")
  │     └─► vector(15) ∥ BM25(15)（并行）→ RRF(k=60) → Cross-Encoder → top 5 chunks
  │
  ├─► [Agent 决策] → 调用 search_knowledge("LangGraph技术原理与适用场景")
  │     └─► vector(15) ∥ BM25(15)（并行）→ RRF(k=60) → Cross-Encoder → top 5 chunks
  │
  ├─► [Agent 决策] → 调用 search_knowledge("LangChain与LangGraph的核心区别对比")
  │     └─► vector(15) ∥ BM25(15)（并行）→ RRF(k=60) → Cross-Encoder → top 5 chunks
  │
  └─► [Agent 生成] 综合所有 15 个 chunks → 结构化回答
  │
  ▼
reflection_node ──▶ 评估回答是否完整准确
  │
  ├── 合格 → END
  └── 不合格 → search_hint → agent_node 重试
```

---

## 6. 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 检索融合 | RRF（非加权求和）| RRF 对排序位置敏感，不要求分数归一化，对异构检索器更鲁棒 |
| 问题分解 | LLM 自动判断 | 避免硬编码规则，LLM 理解语义复杂度 |
| Agent 模式 | LangChain Agent 工具自主调用 | LLM 根据查询语义自主决定先分解还是直接搜索、搜几次、如何综合，比硬编码 if-else 更灵活 |
| 容错降级 | 三层降级兜底 | Agent 失败 → 跳过分解直接检索+生成；检索失败 → 友好提示；确保单点故障不崩整个链路 |
| 向量库 | ChromaDB 持久化 | 支持增量写入，sqlite3 零运维 |
| 图编排 | LangGraph + MemorySaver | 状态持久化、条件分支、checkpoint 回溯 |
| 模型 | DeepSeek V4 + text2vec-base-chinese | 中文场景优化，性价比高 |
| 反思 | Post-hoc 评估 + 关键词优化重试 | 简单有效，比复杂 self-correct 节省 token |
| I/O 并行化 | ThreadPoolExecutor 并行召回 + 并行文档加载 | 向量检索与 BM25 互不依赖，多文档读取互不依赖，并行化可显著降低 I/O 等待延迟 |
| 可插拔知识库 | 代码与领域完全解耦，`knowledge_base/` 目录热插拔 | 替换文档即可切换至法律、医学、金融等任意领域，无需修改任何代码逻辑 |

---

## 7. 演进历程

| 阶段 | V1（加权融合）| V2（RRF+分解）| V3（Agent 模式）| V4（并行化）|
|---|---|---|---|---|
| 召回 | EnsembleRetriever | vector + BM25 独立 | vector + BM25 独立 | **ThreadPoolExecutor 并行** |
| 融合 | 权重 [0.5, 0.5] | RRF (k=60) | RRF (k=60) | RRF (k=60) |
| 分解 | 无 | 硬编码 if-else | LLM Agent 自主决策 | LLM Agent 自主决策 |
| 合成 | RAG 内置生成 | synthesize() | Agent 自主综合 | Agent 自主综合 |
| 决策者 | 代码 | 代码 | **LLM Agent** | LLM Agent |
| 文档加载 | 串行 | 串行 | 串行 | **ThreadPoolExecutor 并行** |
| 灵活性 | 低 | 中 | **高** — Agent 自行决定搜几次 | 高 — Agent 自行决定搜几次 |
| 容错 | 无 | 无 | **三层降级** — Agent→检索→兜底 | 三层降级 — Agent→检索→兜底 |

---

## 8. 文件清单

```
rag-agent/
├── main.py                # 入口：对话循环
├── agent_graph.py          # 编排层：LangGraph 节点 + 图组装
├── rag_engine.py           # 业务层：RRF 融合、检索、问题分解、意图路由
├── vector_store.py         # 基础设施：ChromaDB、vector/BM25 检索器
├── document_loader.py      # 基础设施：文档加载、清洗、分块
├── config.py               # 配置层：LLM、模型、环境变量
├── DESIGN.md               # 本文档：系统设计文档
├── .env                    # API 密钥
├── knowledge_base/         # 知识库源文档（6个 md/pdf）
├── chroma_db_agent/        # 持久化向量库（2564条）
└── .gitignore
```
