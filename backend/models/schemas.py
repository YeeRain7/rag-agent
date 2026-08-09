"""
Pydantic 请求/响应模型
"""

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
