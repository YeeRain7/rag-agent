import os
import re

from concurrent.futures import ThreadPoolExecutor,as_completed

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
    # 第一步：收集全部合法文件路径
    file_paths=[]
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filepath.endswith((".md", ".txt", ".pdf")):
            file_paths.append(filepath)

    all_chunks = []
    # 线程池并发读取所有文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有文件读取任务
        task_futures = [executor.submit(load_single_file, fp) for fp in file_paths]
        #逐个收集结果
        for future in as_completed(task_futures):
            file_chunks = future.result()
            all_chunks.extend(file_chunks)
    # 全局去重
    all_chunks = list(dict.fromkeys(all_chunks))
    print(f"总计加载 {len(all_chunks)} 个有效片段")
    return all_chunks
