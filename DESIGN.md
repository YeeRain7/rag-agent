# Agent 系统设计文档

## 1. 项目概述

**名称**：面向 AI 开发技术文档的 LangGraph 自省式混合召回 RAG 问答 Agent

**核心能力**：
- 意图路由（闲聊 vs 知识查询）
- 问题复杂度判断 + 自动分解子问题
- RRF 多路召回融合（向量 + BM25）
- Cross-Encoder 精排
- 反思自纠错重试机制
- 多轮对话记忆持久化

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
│   │ 意图分流  │─▶│ 分解+检索 │─▶│   反思评估       │   │
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
│   │  意图路由函数      │  │   RAG 检索链路      │        │
│   │  semantic_intent │  │  my_rag_chain      │        │
│   │  llm_router      │  │  my_rag_retrieve   │        │
│   │  CHAT_KW / KNOW_KW│ │  reciprocal_rank_   │        │
│   └─────────────────┘  │  fusion (RRF)      │        │
│                         │  decompose_query   │        │
│   ┌─────────────────┐  │  synthesize         │        │
│   │  问题分解 + 合成   │  └───────────────────┘        │
│   └─────────────────┘                              │
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
  │     ├── config.py (llm)
  │     └── rag_engine.py
  │           ├── config.py (llm, embedding_model, cross_encoder)
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
          ┌────────▼────────┐
          │ 问题复杂度判断     │  decompose_query()
          │ 简单 or 复杂？    │
          └───┬─────────┬───┘
              │         │
         简单  │         │ 复杂
              │         │
    ┌─────────▼──┐  ┌──▼──────────────┐
    │my_rag_chain│  │ 逐子问题检索       │
    │vector+BM25 │  │ for each sub_q:  │
    │  → RRF 15  │  │   RRF → top 5   │
    │  → rerank 5│  │ 合并去重 → rerank │
    │  → generate│  │ synthesize 合成   │
    └──────┬─────┘  └──────┬───────────┘
           │               │
           └───────┬───────┘
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
| RRF top_n（简单）| 15 |
| RRF top_n（每子问题）| 5 |
| 重排 score 阈值 | > 0.25 |
| 重排保留 | top 5 |
| 反思重试上限 | 2 次 |
| 子问题上限 | 4 个 |
| 向量检索 k | 15 |
| BM25 检索 k | 15 |
| 分块大小/重叠 | 500 / 100 |

---

## 5. 数据流（以复杂查询为例）

```
用户: "RAG和微调有什么区别，各适用于什么场景？"
  │
  ▼
router ──▶ 关键词命中 "区别"+"是什么" → 知识查询
  │
  ▼
agent_node ──▶ decompose_query()
  │               LLM 判断: is_complex=True
  │               sub_queries: [
  │                 "RAG技术原理与适用场景",
  │                 "微调技术原理与适用场景",
  │                 "RAG与微调的核心区别对比"
  │               ]
  │
  ├── sub_q1 "RAG技术原理与适用场景"
  │     ├── vector_retriever(15 docs) ─┐
  │     ├── bm25_retriever(15 docs)  ─┤
  │     │                               └── RRF(k=60) → top 5 chunks
  │
  ├── sub_q2 "微调技术原理与适用场景"
  │     ├── vector_retriever(15 docs) ─┐
  │     ├── bm25_retriever(15 docs)  ─┤
  │     │                               └── RRF(k=60) → top 5 chunks
  │
  ├── sub_q3 "RAG与微调的核心区别对比"
  │     ├── vector_retriever(15 docs) ─┐
  │     ├── bm25_retriever(15 docs)  ─┤
  │     │                               └── RRF(k=60) → top 5 chunks
  │
  ├── 收集 15 chunks → dict.fromkeys 去重
  ├── cross_encoder.rerank(原始问题) → top 5
  │
  ▼
synthesize() ──▶ LLM 综合 5 chunks + 原始问题 + 子问题上下文
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
| Agent 模式 | 直接编排（非 LangChain Agent）| 减少一层 LLM 调用，流程更可控、更快 |
| 向量库 | ChromaDB 持久化 | 支持增量写入，sqlite3 零运维 |
| 图编排 | LangGraph + MemorySaver | 状态持久化、条件分支、checkpoint 回溯 |
| 模型 | DeepSeek V4 + text2vec-base-chinese | 中文场景优化，性价比高 |
| 反思 | Post-hoc 评估 + 关键词优化重试 | 简单有效，比复杂 self-correct 节省 token |

---

## 7. 检索链路对比

| 阶段 | V1（旧）| V2（当前）|
|---|---|---|
| 召回 | EnsembleRetriever 加权融合 | vector + BM25 独立检索 |
| 融合 | 权重 [0.5, 0.5] | RRF (k=60) |
| 候选数 | Ensemble 直接出 top 15 | RRF 融合 → top 15 |
| 重排 | Cross-Encoder → top 5 | 不变 |
| 分解 | 无 | LLM 自动判断 + 最多4子问题 |
| 合成 | 无 | 多子问题 chunk 收集 → 去重 → 重排 → 合成 |

---

## 8. 文件清单

```
D:\PythonProject\Agent_test\
├── main.py                # 入口：对话循环
├── agent_graph.py          # 编排层：LangGraph 节点 + 图组装
├── rag_engine.py           # 业务层：RRF、RAG链路、问题分解、意图路由
├── vector_store.py         # 基础设施：ChromaDB、vector/BM25 检索器
├── document_loader.py      # 基础设施：文档加载、清洗、分块
├── config.py               # 配置层：LLM、模型、环境变量
├── DESIGN.md               # 本文档：系统设计文档
├── .env                    # API 密钥
├── knowledge_base/         # 知识库源文档（6个 md/pdf）
├── chroma_db_agent/        # 持久化向量库（2564条）
└── .gitignore
```
