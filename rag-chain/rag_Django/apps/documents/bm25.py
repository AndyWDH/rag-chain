from typing import List, Optional
import pickle
from rank_bm25 import BM25Okapi
from django.conf import settings
import os


def tokenize_chinese(text: str) -> List[str]:
    """中文分词 - 使用 jieba"""
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        return list(text)


class BM25IndexManager:
    _instance = None
    _bm25_index = None
    _corpus = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        pass

    def _ensure_index(self):
        if self._bm25_index is None:
            try:
                from apps.documents.vectorstore import get_vector_store_manager

                vector_manager = get_vector_store_manager()
                vector_manager._ensure_initialized()
                vector_store = vector_manager._vector_store

                if vector_store:
                    all_docs = vector_store.get()
                    if all_docs and 'documents' in all_docs and all_docs['documents']:
                        self._corpus = all_docs['documents']
                        tokenized_corpus = [tokenize_chinese(doc) for doc in self._corpus]
                        self._bm25_index = BM25Okapi(tokenized_corpus)
                        return
            except Exception as e:
                print(f"Failed to build BM25 index from vector store: {e}")

            self._corpus = []
            self._bm25_index = BM25Okapi([])

    def build_index(self, documents: List[str]):
        tokenized_corpus = [tokenize_chinese(doc) for doc in documents]
        self._bm25_index = BM25Okapi(tokenized_corpus)
        self._corpus = documents

    def add_documents(self, documents: List[str]):
        self._ensure_index()
        if not documents:
            return
        tokenized_docs = [tokenize_chinese(doc) for doc in documents]
        for doc in tokenized_docs:
            self._bm25_index.add_tokens(doc)
        self._corpus.extend(documents)

    def get_scores(self, query: str) -> List[float]:
        self._ensure_index()
        if not self._corpus:
            return []
        tokenized_query = tokenize_chinese(query)
        return self._bm25_index.get_scores(tokenized_query)

    def get_top_k(self, query: str, k: int = 10) -> List[tuple]:
        self._ensure_index()
        if not self._corpus:
            return []

        scores = self.get_scores(query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        # 改为允许所有分数 >= 0 的文档
        return [(self._corpus[i], scores[i]) for i in top_indices if scores[i] >= 0]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'bm25_index': self._bm25_index,
                'corpus': self._corpus,
            }, f)

    def load(self, path: str):
        if not os.path.exists(path):
            return

        with open(path, 'rb') as f:
            data = pickle.load(f)
            self._bm25_index = data['bm25_index']
            self._corpus = data['corpus']

    @property
    def corpus(self) -> List[str]:
        return self._corpus or []

    def count(self) -> int:
        return len(self._corpus) if self._corpus else 0


_bm25_manager = BM25IndexManager()


def get_bm25_index() -> Optional[BM25IndexManager]:
    return _bm25_manager


def get_bm25_manager() -> BM25IndexManager:
    return _bm25_manager
