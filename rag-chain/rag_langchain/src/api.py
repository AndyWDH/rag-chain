"""RAG API 服务 - 基于 FastAPI 的高并发服务封装"""

from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pybreaker import CircuitBreaker, CircuitBreakerError

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import RAGPipeline
from src.config import validate_config

# 初始化 FastAPI
app = FastAPI(title="RAG API Service", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 熔断配置
class RAGCircuitBreaker(CircuitBreaker):
    """自定义 RAG 熔断器"""
    def __init__(self):
        super().__init__(
            fail_max=5,      # 连续失败 5 次触发熔断
            reset_timeout=30  # 熔断后 30 秒尝试恢复
        )

rag_circuit_breaker = RAGCircuitBreaker()

# 全局变量
pipeline: Optional[RAGPipeline] = None

# 请求模型
class QueryRequest(BaseModel):
    query: str
    use_cache: bool = True

# 响应模型
class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    cached: bool
    query_class: Optional[str] = None
    class_scores: Optional[Dict[str, float]] = None
    channel_weights: Optional[List[float]] = None
    latency: float = 0.0

class HealthResponse(BaseModel):
    status: str
    message: str
    circuit_breaker_state: str

# 启动时初始化
@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        validate_config()
        pipeline = RAGPipeline()
        pipeline.build_qa_chain()
        print("RAG Pipeline initialized successfully")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        raise

# 熔断保护的查询函数
@rag_circuit_breaker
def protected_query(query: str) -> Dict[str, Any]:
    """受熔断保护的查询函数"""
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return pipeline.query(query)

# 健康检查接口
@app.get("/health", response_model=HealthResponse)
async def health_check():
    state = "closed"
    if rag_circuit_breaker.state == rag_circuit_breaker.STATE_OPEN:
        state = "open"
    elif rag_circuit_breaker.state == rag_circuit_breaker.STATE_HALF_OPEN:
        state = "half-open"
    
    return {
        "status": "healthy" if pipeline else "unhealthy",
        "message": "RAG API Service is running" if pipeline else "Pipeline not initialized",
        "circuit_breaker_state": state
    }

# 查询接口（带熔断保护）
@app.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    """RAG 查询接口"""
    start_time = time.time()
    
    try:
        result = protected_query(body.query)
        
        latency = time.time() - start_time
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            cached=result["cached"],
            query_class=result.get("query_class"),
            class_scores=result.get("class_scores"),
            channel_weights=result.get("channel_weights"),
            latency=round(latency, 3)
        )
    
    except CircuitBreakerError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is temporarily unavailable due to high error rate"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# 批量查询接口
@app.post("/batch_query")
async def batch_query(request: Request, queries: List[str]):
    """批量查询接口"""
    results = []
    for query in queries:
        try:
            result = protected_query(query)
            results.append({
                "query": query,
                "answer": result["answer"],
                "sources": result["sources"],
                "cached": result["cached"]
            })
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e)
            })
    return results

# 统计接口
@app.get("/stats")
async def get_stats():
    """获取服务统计信息"""
    return {
        "circuit_breaker": {
            "state": rag_circuit_breaker.state,
            "failures": rag_circuit_breaker._fail_counter,
            "successes": rag_circuit_breaker._success_counter,
            "failure_threshold": rag_circuit_breaker._fail_max,
            "recovery_timeout": rag_circuit_breaker._reset_timeout
        },
        "pipeline_ready": pipeline is not None,
        "document_count": pipeline.count() if pipeline else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)