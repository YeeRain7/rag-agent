"""
知识库管理 REST 端点 — 对接 ChromaDB
"""

import os
import asyncio
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException

import core

router = APIRouter()

KNOWLEDGE_BASE = os.path.abspath("./knowledge_base")

# ChromaDB SQLite 不支持并发写，用锁串行化
_write_lock = asyncio.Lock()


def _list_files():
    """列出 knowledge_base 中所有支持的文档文件"""
    if not os.path.isdir(KNOWLEDGE_BASE):
        return []
    supported = (".md", ".txt", ".pdf", ".docx")
    files = []
    for f in sorted(os.listdir(KNOWLEDGE_BASE)):
        if f.endswith(supported):
            path = os.path.join(KNOWLEDGE_BASE, f)
            stat = os.stat(path)
            files.append({
                "filename": f,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    return files


@router.get("/documents")
async def list_documents():
    """列出已入库文档"""
    return {"documents": _list_files()}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    上传 PDF/MD/TXT，触发分块 + 向量化 + BM25 重建。
    """
    if core.chromadb_collection is None:
        raise HTTPException(503, "知识库未初始化")

    if not file.filename:
        raise HTTPException(400, "文件名为空")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".md", ".txt", ".pdf", ".docx"):
        raise HTTPException(400, f"不支持的文件格式: {ext}")

    async with _write_lock:
        # 保存文件到 knowledge_base
        os.makedirs(KNOWLEDGE_BASE, exist_ok=True)
        dest = os.path.join(KNOWLEDGE_BASE, file.filename)
        with open(dest, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        # 分块 + 向量化 + 写入 ChromaDB
        from core.vector_store import add_document, rebuild_bm25
        chunk_count = await asyncio.to_thread(add_document, dest)

        # 重建 BM25 索引
        await asyncio.to_thread(rebuild_bm25, KNOWLEDGE_BASE)

        return {
            "success": True,
            "filename": file.filename,
            "chunk_count": chunk_count,
            "message": f"已上传并入库 {chunk_count} 个片段"
        }


@router.delete("/{filename}")
async def delete_document(filename: str):
    """
    删除文档及对应向量。采用全量重建策略（清库 → 重载剩余文档）。
    """
    if core.chromadb_collection is None:
        raise HTTPException(503, "知识库未初始化")

    file_path = os.path.join(KNOWLEDGE_BASE, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(404, f"文档不存在: {filename}")

    async with _write_lock:
        os.remove(file_path)

        # 只删除该文件的 chunks（用 metadata 定位），不全量重建
        await asyncio.to_thread(_delete_chunks_by_source, filename)

        from core.vector_store import rebuild_bm25
        await asyncio.to_thread(rebuild_bm25, KNOWLEDGE_BASE)

        return {"success": True, "message": f"文档 {filename} 已删除"}


@router.get("/stats")
async def knowledge_stats():
    """知识库统计"""
    files = _list_files()
    chunk_count = core.chromadb_collection.count() if core.chromadb_collection is not None else 0
    return {
        "document_count": len(files),
        "total_chunks": chunk_count
    }


def _delete_chunks_by_source(filename: str):
    """
    根据 metadata 中的 source 字段，删除指定文件的全部 chunk。
    在写入锁内调用。
    """
    try:
        result = core.chromadb_collection.get(
            where={"source": filename},
            limit=core.chromadb_collection.count() + 100
        )
        ids_to_delete = result["ids"]
        if ids_to_delete:
            core.chromadb_collection.delete(ids=ids_to_delete)
            print(f"[OK] 已删除 {filename} 的 {len(ids_to_delete)} 个 chunk")
        else:
            print(f"[WARN] 未找到 {filename} 的 chunk（可能无元数据），跳过")
    except Exception as e:
        print(f"[WARN] 按 source 删除失败: {e}，跳过向量删除")
