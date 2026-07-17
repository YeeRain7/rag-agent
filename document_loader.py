import os
import re
import fitz  # PyMuPDF

from config import text_splitter


def clean_markdown(text: str) -> str:
    """清洗Markdown标记：移除图片、链接、规范化空白"""
    # 1. 移除图片（RAG看不了图，全删掉）
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'<img.*?>', '', text)

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


def load_all_docs(folder_path: str) -> list[str]:
    """加载文件夹中所有支持的文档，清洗、分块、去重"""
    all_chunks = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        raw_text = ""
        # 分支读取不同格式文件
        if filename.endswith((".md", ".txt")):
            with open(filepath, 'r', encoding="utf-8") as f:
                raw_text = f.read()
        elif filename.endswith(".pdf"):
            raw_text = load_pdf(filepath)
        else:
            continue  # 跳过不支持的文件类型

        # 1. 清洗markdown标记
        clean_text = clean_markdown(raw_text)
        # 2. 语义分块
        chunks = text_splitter.split_text(clean_text)
        # 过滤空片段
        chunks = [c.strip() for c in chunks if c.strip()]
        # 加入全局列表
        all_chunks.extend(chunks)

    # 全局去重
    all_chunks = list(dict.fromkeys(all_chunks))
    print(f"总计加载 {len(all_chunks)} 个有效片段")
    return all_chunks
