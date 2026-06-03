import hashlib
import time
from typing import Dict, List, Optional
from django.core.cache import cache
from django.conf import settings
from apps.retrieval.graph import RetrievalGraph


def _init_llm():
    """初始化 LLM 客户端 - 参考 LangChain 框架实现"""
    try:
        from langchain_openai import ChatOpenAI
        
        # 检查 API Key 是否配置
        api_key = settings.DASHSCOPE_API_KEY
        if not api_key:
            print("警告: DASHSCOPE_API_KEY 未配置，LLM 功能将不可用")
            return None
        
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.3,
            max_tokens=2048,
        )
        print(f"LLM 初始化成功: {settings.LLM_MODEL}")
        return llm
    except Exception as e:
        print(f"LLM 初始化失败: {e}")
        return None


class RetrievalService:
    _instance = None
    _graph = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        pass

    def _initialize_graph(self):
        if self._graph is not None:
            return
            
        try:
            from apps.documents.vectorstore import get_vector_store, get_vector_store_manager
            from apps.documents.bm25 import get_bm25_index

            # 确保向量存储初始化
            vector_manager = get_vector_store_manager()
            vector_manager._ensure_initialized()
            vector_store = vector_manager._vector_store
            
            bm25_index = get_bm25_index()
            
            # 初始化 LLM
            llm = _init_llm()

            print(f"Vector store initialized: {vector_store is not None}")
            print(f"BM25 index initialized: {bm25_index is not None}")
            print(f"LLM initialized: {llm is not None}")

            self._graph = RetrievalGraph(
                vector_store=vector_store,
                bm25_index=bm25_index,
                llm=llm
            )
        except Exception as e:
            print(f"Failed to initialize retrieval graph: {e}")
            import traceback
            traceback.print_exc()
            self._graph = RetrievalGraph()

    def _ensure_initialized(self):
        if self._graph is None:
            self._initialize_graph()

    def _get_cache_key(self, query: str) -> str:
        return f"query_result:{hashlib.md5(query.encode()).hexdigest()}"

    def query(self, query: str, use_cache: bool = True, collection_id: int = None) -> Dict:
        start_time = time.time()

        if use_cache:
            cache_key = self._get_cache_key(query)
            cached_result = cache.get(cache_key)
            if cached_result:
                cached_result['cached'] = True
                return cached_result

        self._ensure_initialized()
        
        try:
            result = self._graph.invoke(query)

            latency = time.time() - start_time
            result['latency'] = latency
            result['cached'] = False

            if use_cache and not result.get('error'):
                cache_ttl = settings.CACHE_TTL.get('query_result', 300)
                cache.set(self._get_cache_key(query), result, cache_ttl)

            return result

        except Exception as e:
            return {
                'query': query,
                'answer': f"查询出错: {str(e)}",
                'sources': [],
                'query_class': 'balanced',
                'class_scores': {},
                'channel_weights': [0.5, 0.5],
                'latency': time.time() - start_time,
                'cached': False,
                'error': str(e),
            }

    async def aquery(self, query: str, use_cache: bool = True, collection_id: int = None) -> Dict:
        start_time = time.time()

        if use_cache:
            cache_key = self._get_cache_key(query)
            cached_result = cache.get(cache_key)
            if cached_result:
                cached_result['cached'] = True
                return cached_result

        self._ensure_initialized()
        
        try:
            result = await self._graph.ainvoke(query)

            latency = time.time() - start_time
            result['latency'] = latency
            result['cached'] = False

            if use_cache and not result.get('error'):
                cache_ttl = settings.CACHE_TTL.get('query_result', 300)
                cache.set(self._get_cache_key(query), result, cache_ttl)

            return result

        except Exception as e:
            return {
                'query': query,
                'answer': f"查询出错: {str(e)}",
                'sources': [],
                'query_class': 'balanced',
                'class_scores': {},
                'channel_weights': [0.5, 0.5],
                'latency': time.time() - start_time,
                'cached': False,
                'error': str(e),
            }


retrieval_service = RetrievalService()


def get_retrieval_service() -> RetrievalService:
    return retrieval_service