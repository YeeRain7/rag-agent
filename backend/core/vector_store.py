import os

import chromadb
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from .config import embedding_model, EmbedChunk
from .document_loader import load_all_docs, load_all_docs_with_meta

# 模块级变量（延迟初始化，由 FastAPI startup 事件调用 init_vector_store() 设置）
chromadb_client = None
chromadb_collection = None
vector_store = None
vector_retriever = None
bm25_retriever = None


def init_vector_store(knowledge_base_path: str = "./knowledge_base"):
    """
    初始化向量库和检索器。
    由 FastAPI startup 事件调用，避免 import 时自动加载模型。

    Returns:
        dict: 包含 vector_retriever, bm25_retriever, chromadb_collection,
              chromadb_client, vector_store 的字典
    """
    global chromadb_client, chromadb_collection, vector_store
    global vector_retriever, bm25_retriever

    # 1. 初始化向量库
    chromadb_client = chromadb.PersistentClient(path="./chroma_db_agent")
    chromadb_collection = chromadb_client.get_or_create_collection(name="default")

    # 2. 加载文档（无论库空不空，先加载用于BM25）
    chunks = load_all_docs(knowledge_base_path)

    # 3. 条件嵌入入库：仅空库时写入，避免重复插入
    if chromadb_collection.count() == 0:
        print("向量库为空，正在加载文档并持久化...")
        chunks_with_meta, metadatas = load_all_docs_with_meta(knowledge_base_path)
        print(f"一共 {len(chunks_with_meta)} 个片段，开始批量编码向量")
        embeddings = embedding_model.encode(chunks_with_meta, batch_size=16).tolist()
        ids = [str(i) for i in range(len(chunks_with_meta))]

        batch_size = 100
        total = len(chunks_with_meta)
        for start in range(0, total, batch_size):
            end = start + batch_size
            chromadb_collection.add(
                documents=chunks_with_meta[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end]
            )
            print(f"[OK] 已写入 {min(end, total)} / {total}")
        print("[OK] 全部入库完成！")
    else:
        print(f"\n已加载持久化向量库，现有文档总数：{chromadb_collection.count()}")

    # 4. 构建检索器
    vector_store = Chroma(
        client=chromadb_client,
        collection_name="default",
        embedding_function=EmbedChunk()
    )

    # 向量检索器
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # BM25检索器
    bm25_retriever = BM25Retriever.from_texts(chunks, k=20)

    print("检索器已就绪：vector_retriever + bm25_retriever 供 rag_engine 做 RRF 融合")

    return {
        "vector_retriever": vector_retriever,
        "bm25_retriever": bm25_retriever,
        "chromadb_collection": chromadb_collection,
        "chromadb_client": chromadb_client,
        "vector_store": vector_store,
    }


def add_document(file_path: str) -> int:
    """
    增量添加单个文档：分块 → 编码 → 写入 ChromaDB。
    """
    import core as _core
    from .document_loader import load_single_file

    col = _core.chromadb_collection
    if col is None:
        raise RuntimeError("向量库未初始化")

    chunks = load_single_file(file_path)
    if not chunks:
        print(f"[WARN] 文件 {file_path} 未能提取到有效片段")
        return 0

    src = os.path.basename(file_path)
    ext = os.path.splitext(src)[1].lower()
    doc_type = {".pdf": "PDF", ".md": "Markdown", ".txt": "TXT", ".docx": "Word"}.get(ext, ext)
    metas = [{"source": src, "type": doc_type} for _ in chunks]

    embeddings = embedding_model.encode(chunks, batch_size=16).tolist()
    existing_count = col.count()
    ids = [str(existing_count + i) for i in range(len(chunks))]

    batch_size = 100
    total = len(chunks)
    for start in range(0, total, batch_size):
        end = start + batch_size
        col.add(
            documents=chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=metas[start:end],
            ids=ids[start:end]
        )

    print(f"[OK] 新增文档 {src}，{total} 个 chunk 已入库")
    return total


def rebuild_bm25(knowledge_base_path: str = "./knowledge_base"):
    """
    重建 BM25 检索器（上传/删除文档后调用）。

    重新加载全部文档，构建新的 BM25Retriever，
    并同步更新 core 包的单例引用。
    """
    global bm25_retriever

    from .document_loader import load_all_docs

    chunks = load_all_docs(knowledge_base_path)
    bm25_retriever = BM25Retriever.from_texts(chunks, k=20)

    # 同步更新 core 包的单例
    import core as _core
    _core.bm25_retriever = bm25_retriever

    print(f"[OK] BM25 索引已重建，共 {len(chunks)} 个片段")
