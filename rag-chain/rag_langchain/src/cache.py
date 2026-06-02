"""缓存层实现 - 支持三级缓存"""

import hashlib
import json
from typing import Any, Optional, List

try:
    import redis
    from redis.asyncio import Redis as AsyncRedis
except ImportError:
    redis = None
    AsyncRedis = None

from src.config import CACHE_ENABLED, REDIS_URL, CACHE_TTL_RESULT, CACHE_TTL_RETRIEVAL, CACHE_TTL_EMBEDDING


class CacheManager:
    """缓存管理器 - 支持结果缓存、检索缓存、向量缓存"""
    
    def __init__(self):
        self.enabled = CACHE_ENABLED
        self.client = None
        self.async_client = None
        if self.enabled and redis:
            try:
                self.client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
                self.async_client = AsyncRedis.from_url(REDIS_URL, decode_responses=True)
                self.client.ping()
                print("Redis 缓存连接成功")
            except Exception as e:
                print(f"Redis 连接失败，将使用内存缓存: {e}")
                self.enabled = False
                self.client = None
                self.async_client = None
        
        # 内存缓存作为降级方案
        self._memory_cache = {}
    
    def _get_key(self, prefix: str, content: str) -> str:
        """生成缓存键"""
        return f"{prefix}:{hashlib.md5(content.encode()).hexdigest()}"
    
    def get_result(self, query: str) -> Optional[str]:
        """获取问答结果缓存"""
        if not self.enabled:
            return self._memory_cache.get(f"q:{query}")
        
        key = self._get_key("q", query)
        try:
            return self.client.get(key)
        except Exception:
            return None
    
    async def aget_result(self, query: str) -> Optional[str]:
        """异步获取问答结果缓存"""
        if not self.enabled:
            return self._memory_cache.get(f"q:{query}")
        
        key = self._get_key("q", query)
        try:
            return await self.async_client.get(key)
        except Exception:
            return None
    
    def set_result(self, query: str, answer: str) -> None:
        """设置问答结果缓存"""
        if not self.enabled:
            self._memory_cache[f"q:{query}"] = answer
            return
        
        key = self._get_key("q", query)
        try:
            self.client.setex(key, CACHE_TTL_RESULT, answer)
        except Exception:
            pass
    
    async def aset_result(self, query: str, answer: str) -> None:
        """异步设置问答结果缓存"""
        if not self.enabled:
            self._memory_cache[f"q:{query}"] = answer
            return
        
        key = self._get_key("q", query)
        try:
            await self.async_client.setex(key, CACHE_TTL_RESULT, answer)
        except Exception:
            pass
    
    def get_retrieval(self, query: str) -> Optional[List[str]]:
        """获取检索结果缓存"""
        if not self.enabled:
            return self._memory_cache.get(f"r:{query}")
        
        key = self._get_key("r", query)
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None
    
    async def aget_retrieval(self, query: str) -> Optional[List[str]]:
        """异步获取检索结果缓存"""
        if not self.enabled:
            return self._memory_cache.get(f"r:{query}")
        
        key = self._get_key("r", query)
        try:
            data = await self.async_client.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None
    
    def set_retrieval(self, query: str, doc_ids: List[str]) -> None:
        """设置检索结果缓存"""
        if not self.enabled:
            self._memory_cache[f"r:{query}"] = doc_ids
            return
        
        key = self._get_key("r", query)
        try:
            self.client.setex(key, CACHE_TTL_RETRIEVAL, json.dumps(doc_ids))
        except Exception:
            pass
    
    async def aset_retrieval(self, query: str, doc_ids: List[str]) -> None:
        """异步设置检索结果缓存"""
        if not self.enabled:
            self._memory_cache[f"r:{query}"] = doc_ids
            return
        
        key = self._get_key("r", query)
        try:
            await self.async_client.setex(key, CACHE_TTL_RETRIEVAL, json.dumps(doc_ids))
        except Exception:
            pass
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """获取向量缓存"""
        if not self.enabled:
            return self._memory_cache.get(f"e:{text}")
        
        key = self._get_key("e", text)
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None
    
    async def aget_embedding(self, text: str) -> Optional[List[float]]:
        """异步获取向量缓存"""
        if not self.enabled:
            return self._memory_cache.get(f"e:{text}")
        
        key = self._get_key("e", text)
        try:
            data = await self.async_client.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None
    
    def set_embedding(self, text: str, embedding: List[float]) -> None:
        """设置向量缓存"""
        if not self.enabled:
            self._memory_cache[f"e:{text}"] = embedding
            return
        
        key = self._get_key("e", text)
        try:
            self.client.setex(key, CACHE_TTL_EMBEDDING, json.dumps(embedding))
        except Exception:
            pass
    
    async def aset_embedding(self, text: str, embedding: List[float]) -> None:
        """异步设置向量缓存"""
        if not self.enabled:
            self._memory_cache[f"e:{text}"] = embedding
            return
        
        key = self._get_key("e", text)
        try:
            await self.async_client.setex(key, CACHE_TTL_EMBEDDING, json.dumps(embedding))
        except Exception:
            pass


# 全局缓存实例
cache_manager = CacheManager()
