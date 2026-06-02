"""配置管理 - 集中管理所有参数"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── LLM 配置 ──────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ─── Embedding 配置 ────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# ─── 向量库配置 ────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")

# ─── 文本切分配置 ──────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ─── 检索配置 ──────────────────────────────────────────────────────────
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

# ─── 缓存配置 ──────────────────────────────────────────────────────────
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL_RESULT = int(os.getenv("CACHE_TTL_RESULT", "300"))      # 5分钟
CACHE_TTL_RETRIEVAL = int(os.getenv("CACHE_TTL_RETRIEVAL", "1800")) # 30分钟
CACHE_TTL_EMBEDDING = int(os.getenv("CACHE_TTL_EMBEDDING", "86400")) # 24小时

# ─── 并发配置 ──────────────────────────────────────────────────────────
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "50"))
LLM_POOL_SIZE = int(os.getenv("LLM_POOL_SIZE", "10"))

# ─── 路径配置 ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_docs")


def validate_config() -> None:
    """启动时校验关键配置"""
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY.startswith("sk-your-key"):
        raise ValueError(
            "\n缺少 DASHSCOPE_API_KEY\n"
            "1) 复制配置模板: cp .env.example .env\n"
            "2) 编辑 .env,填入你的 DashScope API key\n"
            "3) 申请地址: https://dashscope.console.aliyuncs.com/apiKey\n"
        )
