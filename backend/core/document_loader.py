import os
import re

from concurrent.futures import ThreadPoolExecutor,as_completed

import fitz  # PyMuPDF

from .config import text_splitter


def clean_markdown(text: str) -> str:
    """清洗Markdown标记：移除图片、链接、规范化空白"""
    # 1. 移除图片（RAG看不了图，全删掉）
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'<img.*?>', '', text)
    text = re.sub(r'<Image.*?>', '', text)

    # 2. 移除网页链接，只保留链接文字
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # 3. 统一换行符：多个连续空行合并成两个换行
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # 4. 移除行尾空格和过长空白
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


def load_pdf(filepath: str) -> str:
    """从PDF中提取纯文本"""
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def load_docx(filepath: str) -> str:
    """从 DOCX 中提取纯文本（段落）"""
    from docx import Document
    doc = Document(filepath)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_single_file(file_path: str) -> list[str]:
    """加载文件夹中所有支持的文档，清洗、分块、去重"""
    try:
        filename=os.path.basename(file_path)
        raw_text = ""
        # 分支读取不同格式文件
        if filename.endswith((".md", ".txt")):
            with open(file_path, 'r', encoding="utf-8") as f:
                raw_text = f.read()
        elif filename.endswith(".pdf"):
            raw_text = load_pdf(file_path)
        elif filename.endswith(".docx"):
            raw_text = load_docx(file_path)
        else:
            return []
        # 1. 清洗markdown标记
        clean_text = clean_markdown(raw_text)
        # 2. 语义分块
        chunks = text_splitter.split_text(clean_text)
        # 过滤空片段
        chunks = [c.strip() for c in chunks if c.strip()]
        print(f"已加载 {filename}，得到 {len(chunks)} 个语义片段")
        return chunks
    except Exception as e:
        print(f"警告：文件 {file_path} 加载失败，跳过，异常：{str(e)}")
        return []

def load_all_docs(folder_path: str, max_workers: int = 4) -> list[str]:
    """并行加载文件夹中所有支持的文档，清洗、分块、去重"""
    file_paths = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filepath.endswith((".md", ".txt", ".pdf", ".docx")):
            file_paths.append(filepath)

    all_chunks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        task_futures = [executor.submit(load_single_file, fp) for fp in file_paths]
        for future in as_completed(task_futures):
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"警告：加载文件时发生异常：{str(e)}")
    all_chunks = list(dict.fromkeys(all_chunks))
    print(f"总计加载 {len(all_chunks)} 个有效片段")
    return all_chunks


def load_all_docs_with_meta(folder_path: str, max_workers: int = 4):
    """
    并行加载并返回结构化数据：chunks 文本 + 每块的来源元数据。
    Returns: (chunks: list[str], metadatas: list[dict])
      每个 metadata: {source: 文件名, type: pdf|md|txt}
    """
    file_paths = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filepath.endswith((".md", ".txt", ".pdf", ".docx")):
            file_paths.append(filepath)

    source_name_map = {}  # fp → basename
    for fp in file_paths:
        source_name_map[fp] = os.path.basename(fp)

    def _load_with_source(fp):
        chunks = load_single_file(fp)
        src = source_name_map[fp]
        ext = os.path.splitext(src)[1].lower()
        doc_type = {".pdf": "PDF", ".md": "Markdown", ".txt": "TXT", ".docx": "Word"}.get(ext, ext)
        metas = [{"source": src, "type": doc_type} for _ in chunks]
        return chunks, metas

    all_chunks = []
    all_metas = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_load_with_source, fp): fp for fp in file_paths}
        for future in as_completed(futures):
            try:
                chunks, metas = future.result()
            except Exception as e:
                print(f"警告：加载文件时发生异常：{str(e)}")
                continue
            # 去重
            seen = set(all_chunks)
            for c, m in zip(chunks, metas):
                if c not in seen:
                    seen.add(c)
                    all_chunks.append(c)
                    all_metas.append(m)

    print(f"总计加载 {len(all_chunks)} 个有效片段（含元数据）")
    return all_chunks, all_metas
