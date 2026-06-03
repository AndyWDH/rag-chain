from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
try:
    from langchain.schema import Embeddings
except ImportError:
    from langchain.embeddings.base import Embeddings
from django.conf import settings
import os


class DashScopeEmbeddings(Embeddings):
    """自定义 DashScope Embedding 包装器 - 与 LangChain 框架保持一致"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v2"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client = None
    
    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        client = self._ensure_client()
        embeddings = []
        batch_size = 25
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(model=self.model, input=batch)
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        client = self._ensure_client()
        response = client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


class VectorStoreManager:
    _instance = None
    _vector_store = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        pass

    def _initialize_vector_store(self):
        if self._vector_store is not None:
            return
        
        try:
            persist_dir = settings.CHROMA_PERSIST_DIR
            os.makedirs(persist_dir, exist_ok=True)

            # 使用与 LangChain 框架相同的 DashScope Embeddings
            embedding = DashScopeEmbeddings(
                api_key=settings.DASHSCOPE_API_KEY or settings.OPENAI_API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model="text-embedding-v2"
            )

            self._vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding
            )
        except Exception as e:
            print(f"Failed to initialize vector store: {e}")
            self._vector_store = None

    def _ensure_initialized(self):
        if self._vector_store is None:
            self._initialize_vector_store()

    def add_documents(self, texts: List[str], metadatas: List[dict] = None, ids: List[str] = None):
        self._ensure_initialized()
        if self._vector_store is None:
            raise RuntimeError("Vector store not initialized")

        return self._vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def similarity_search(self, query: str, k: int = 10, filter: dict = None):
        self._ensure_initialized()
        if self._vector_store is None:
            return []

        return self._vector_store.similarity_search(query, k=k, filter=filter)

    def similarity_search_by_vector(self, query_vector: List[float], k: int = 10, filter: dict = None):
        self._ensure_initialized()
        if self._vector_store is None:
            return []

        return self._vector_store.similarity_search_by_vector(query_vector, k=k, filter=filter)

    def delete(self, ids: List[str] = None, delete_all: bool = False):
        self._ensure_initialized()
        if self._vector_store is None:
            return

        if delete_all:
            self._vector_store.delete_collection()
        else:
            self._vector_store.delete(ids)

    def count(self) -> int:
        self._ensure_initialized()
        if self._vector_store is None:
            return 0
        return self._vector_store._collection.count()


_vector_store_manager = VectorStoreManager()


def get_vector_store() -> Optional[Chroma]:
    # 不在这里调用 _ensure_initialized()，让调用者决定何时初始化
    return _vector_store_manager._vector_store


def get_vector_store_manager() -> VectorStoreManager:
    return _vector_store_manager