from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document


class RetrievalState(TypedDict):
    query: str
    query_class: str
    class_scores: Dict[str, float]
    query_vector: Optional[List[float]]
    channel_weights: List[float]
    retrieved_docs: List[Document]
    bm25_results: List[Document]
    merged_docs: List[Document]
    reranked_docs: List[Document]
    answer: str
    sources: List[str]
    metadata: Dict[str, Any]
    error: Optional[str]
    cached: bool


class QueryClassificationState(TypedDict):
    query: str
    query_class: str
    class_scores: Dict[str, float]


class RetrievalResultState(TypedDict):
    vector_results: List[Document]
    bm25_results: List[Document]
    merged_results: List[Document]


class RerankState(TypedDict):
    docs: List[Document]
    reranked_docs: List[Document]


class GenerationState(TypedDict):
    query: str
    context: str
    answer: str
    sources: List[str]