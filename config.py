import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder

# 物理核心8核
cpu_core_count = 8
os.environ["OMP_NUM_THREADS"] = str(cpu_core_count)
os.environ["MKL_NUM_THREADS"] = str(cpu_core_count)
torch.set_num_threads(cpu_core_count)
torch.set_num_interop_threads(2)   # interop线程保持少量，负责并行任务调度，不建议开大

load_dotenv()

# 1. LLM 初始化
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("api_key"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.2
)

# 2. 向量模型、重排模型全局加载
embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

# 3. 全局统一分块器（只初始化一次）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "，", " "]
)

# 4. 切片向量化包装类（兼容 LangChain 的 embed_query / embed_documents 接口）
class EmbedChunk:
    def embed_query(self, text: str) -> list[float]:
        return embedding_model.encode(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embedding_model.encode(texts).tolist()
