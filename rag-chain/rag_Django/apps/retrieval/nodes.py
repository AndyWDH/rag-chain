import re
from typing import List, Dict
from langchain_core.documents import Document
from django.conf import settings


class QueryClassifierNode:
    DOMAIN_KEYWORDS = {
        'policy': ['保单', '保险', '条款', '合同', '险种', '保费', '保额', '理赔', '赔付'],
        'health': ['疾病', '医疗', '住院', '手术', '门诊', '体检', '健康', '治疗'],
        'accident': ['意外', '伤残', '身故', '烧伤', '烫伤', '骨折', '意外事故'],
        'finance': ['缴费', '缴费期', '等待期', '犹豫期', '宽限期', '现金价值', '红利'],
        'liability': ['责任', '免责', '除外', '不保', '拒绝', '不予赔付'],
    }

    QUESTION_PATTERNS = [
        r'^什么是',
        r'^如何',
        r'^多少',
        r'^哪里',
        r'^怎么',
        r'^为什么',
        r'^能否',
        r'^是否',
        r'^可以',
        r'^请问',
        r'吗\?$',
        r'\?$',
    ]

    def __init__(self):
        self.question_regex = [re.compile(p, re.IGNORECASE) for p in self.QUESTION_PATTERNS]

    def extract_features(self, query: str) -> Dict[str, any]:
        query_len = len(query)
        has_question = any(pattern.search(query) for pattern in self.question_regex)
        domain_scores = {}

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query)
            domain_scores[domain] = score

        dominant_domain = max(domain_scores, key=domain_scores.get) if any(domain_scores.values()) else None

        keyword_count = sum(domain_scores.values())

        return {
            'length': query_len,
            'has_question': has_question,
            'domain_scores': domain_scores,
            'dominant_domain': dominant_domain,
            'keyword_count': keyword_count,
        }

    def calculate_class_scores(self, features: Dict) -> Dict[str, float]:
        keyword_score = 0.0
        semantic_score = 0.0
        balanced_score = 0.0

        if features['length'] <= 15:
            keyword_score += 0.4
        elif features['length'] <= 30:
            keyword_score += 0.2
            balanced_score += 0.2
        else:
            semantic_score += 0.3
            balanced_score += 0.2

        if features['has_question']:
            semantic_score += 0.4
        else:
            keyword_score += 0.3

        if features['keyword_count'] >= 3:
            keyword_score += 0.3
        elif features['keyword_count'] >= 1:
            balanced_score += 0.3
        else:
            semantic_score += 0.3

        total = keyword_score + semantic_score + balanced_score
        if total > 0:
            keyword_score /= total
            semantic_score /= total
            balanced_score /= total

        return {
            'keyword': round(keyword_score, 2),
            'semantic': round(semantic_score, 2),
            'balanced': round(balanced_score, 2),
        }

    def classify(self, state: Dict) -> Dict:
        query = state['query']
        features = self.extract_features(query)
        class_scores = self.calculate_class_scores(features)

        query_class = max(class_scores, key=class_scores.get)
        weights = settings.CHANNEL_WEIGHTS.get(query_class, [0.5, 0.5])

        state['query_class'] = query_class
        state['class_scores'] = class_scores
        state['channel_weights'] = weights

        return state


class VectorRetrieverNode:
    def __init__(self, vector_store=None):
        self.vector_store = vector_store

    def retrieve(self, state: Dict) -> Dict:
        if not self.vector_store:
            state['retrieved_docs'] = []
            return state

        try:
            query = state.get('query', '')
            if not query:
                state['retrieved_docs'] = []
                return state
            
            docs = self.vector_store.similarity_search(
                query,
                k=settings.RETRIEVAL_K
            )
            state['retrieved_docs'] = docs
        except Exception as e:
            state['retrieved_docs'] = []
            state['error'] = f"Vector retrieval error: {str(e)}"

        return state


class BM25RetrieverNode:
    def __init__(self, bm25_index=None):
        self.bm25_index = bm25_index

    def retrieve(self, state: Dict) -> Dict:
        if not self.bm25_index:
            state['bm25_results'] = []
            return state

        try:
            query = state['query']
            doc_scores = self.bm25_index.get_scores(query)
            
            # 调试信息
            print(f"BM25 query: {query}")
            print(f"BM25 scores: {doc_scores}")
            
            top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:settings.RETRIEVAL_K]
            # 放宽过滤条件，允许分数大于等于0的文档
            docs = [self.bm25_index.corpus[i] for i in top_indices if doc_scores[i] >= 0]
            state['bm25_results'] = [Document(page_content=d) for d in docs]
            
            print(f"BM25 results count: {len(state['bm25_results'])}")
        except Exception as e:
            state['bm25_results'] = []
            state['error'] = f"BM25 retrieval error: {str(e)}"
            print(f"BM25 error: {e}")

        return state


class RRFMergerNode:
    def __init__(self, k: int = 60):
        self.k = k

    def merge(self, state: Dict) -> Dict:
        vector_docs = state.get('retrieved_docs', [])
        bm25_docs = state.get('bm25_results', [])
        weights = state.get('channel_weights', [0.5, 0.5])

        if not vector_docs and not bm25_docs:
            state['merged_docs'] = []
            return state

        doc_scores = {}

        for rank, doc in enumerate(vector_docs):
            key = doc.page_content[:100]
            score = (1 / (self.k + rank + 1)) * weights[0]
            doc_scores[key] = doc_scores.get(key, 0) + score

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:100]
            score = (1 / (self.k + rank + 1)) * weights[1]
            doc_scores[key] = doc_scores.get(key, 0) + score

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        merged = []
        seen = set()

        for key, score in sorted_docs:
            for doc in vector_docs + bm25_docs:
                if doc.page_content[:100] == key and key not in seen:
                    merged.append(doc)
                    seen.add(key)
                    break

        state['merged_docs'] = merged[:settings.RETRIEVAL_K]
        return state


class CrossEncoderRerankerNode:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                raise RuntimeError(f"Failed to load CrossEncoder: {e}")
        return self._model

    def rerank(self, state: Dict) -> Dict:
        docs = state.get('merged_docs', [])

        if len(docs) <= settings.RERANK_TOP_N:
            state['reranked_docs'] = docs
            return state

        try:
            model = self._ensure_model()
            query = state['query']
            pairs = [(query, doc.page_content) for doc in docs]
            scores = model.predict(pairs)

            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            state['reranked_docs'] = [doc for doc, score in scored_docs[:settings.RERANK_TOP_N]]
        except Exception as e:
            state['reranked_docs'] = docs[:settings.RERANK_TOP_N]
            state['error'] = f"Rerank error: {str(e)}"

        return state


class LLMGeneratorNode:
    def __init__(self, llm=None):
        self.llm = llm

    def generate(self, state: Dict) -> Dict:
        query = state['query']
        docs = state.get('reranked_docs', [])

        if not docs:
            state['answer'] = "抱歉，未找到相关文档来回答您的问题。"
            state['sources'] = []
            return state

        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = self._build_prompt(query, context)

        try:
            if self.llm:
                answer = self.llm.invoke(prompt)
                state['answer'] = answer.content if hasattr(answer, 'content') else str(answer)
            else:
                state['answer'] = f"基于{len(docs)}篇相关文档生成回答（LLM未配置）"
        except Exception as e:
            state['answer'] = f"生成回答时出错: {str(e)}"

        state['sources'] = [doc.metadata.get('source', 'unknown') for doc in docs if doc.metadata]

        return state

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""基于以下参考资料回答问题。如果资料不足，请说明。

参考资料：
{context}

问题：{query}

回答："""