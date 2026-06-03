from typing import Dict


def classify_to_retrieve(state: Dict) -> str:
    query_class = state.get('query_class', 'balanced')
    return query_class


def route_by_query_class(state: Dict) -> str:
    query_class = state.get('query_class', 'balanced')

    if query_class == 'keyword':
        return 'bm25_retriever'
    elif query_class == 'semantic':
        return 'vector_retriever'
    else:
        return 'both_retrievers'


def after_retrieval(state: Dict) -> str:
    if state.get('error'):
        return 'error_handler'
    return 'rrf_merger'


def after_rrf(state: Dict) -> str:
    docs = state.get('merged_docs', [])
    if not docs:
        return 'generate'
    return 'reranker'


def after_rerank(state: Dict) -> str:
    return 'generate'


def error_handler(state: Dict) -> str:
    return 'generate'