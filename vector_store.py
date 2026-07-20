import chromadb
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from config import embedding_model, EmbedChunk
from document_loader import load_all_docs

# 1. 初始化向量库
chromadb_client = chromadb.PersistentClient(path="./chroma_db_agent")
chromadb_collection = chromadb_client.get_or_create_collection(name="default")

# 2. 加载文档（无论库空不空，先加载用于BM25）
chunks = load_all_docs("./knowledge_base")

# 3. 条件嵌入入库：仅空库时写入，避免重复插入
if chromadb_collection.count() == 0:
    print("向量库为空，正在加载文档并持久化...")
    print(f"一共 {len(chunks)} 个片段，开始批量编码向量")
    # 批量向量化
    embeddings = embedding_model.encode(chunks, batch_size=16).tolist()
    ids = [str(i) for i in range(len(chunks))]

    # 小批次写入，每批100条，降低sqlite压力
    batch_size = 100
    total = len(chunks)
    for start in range(0, total, batch_size):
        end = start + batch_size
        chromadb_collection.add(
            documents=chunks[start:end],
            embeddings=embeddings[start:end],
            ids=ids[start:end]
        )
        print(f"✅ 已写入 {min(end, total)} / {total}")
    print("🎉 全部入库完成！")
else:
    print(f"\n已加载持久化向量库，现有文档总数：{chromadb_collection.count()}")

# 4. 构建检索器
vector_store = Chroma(
    client=chromadb_client,
    collection_name="default",
    embedding_function=EmbedChunk()
)

# 向量检索器
vector_retriever = vector_store.as_retriever()

# BM25检索器
bm25_retriever = BM25Retriever.from_texts(chunks)

# 检索器已就绪：vector_retriever + bm25_retriever 供 rag_engine 做 RRF 融合
