from typing import Dict
from langgraph.graph import StateGraph, END
from apps.retrieval.states import RetrievalState
from apps.retrieval.nodes import (
    QueryClassifierNode,
    VectorRetrieverNode,
    BM25RetrieverNode,
    RRFMergerNode,
    CrossEncoderRerankerNode,
    LLMGeneratorNode,
)


class RetrievalGraph:
    def __init__(self, vector_store=None, bm25_index=None, llm=None):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.llm = llm

        self.classifier = QueryClassifierNode()
        self.vector_retriever = VectorRetrieverNode(vector_store)
        self.bm25_retriever = BM25RetrieverNode(bm25_index)
        self.rrf_merger = RRFMergerNode()
        self.reranker = CrossEncoderRerankerNode()
        self.generator = LLMGeneratorNode(llm)

        self.graph = self._build_graph()

    def _route_decision(self, state: Dict) -> str:
        query_class = state.get('query_class', 'balanced')
        if query_class == 'keyword':
            return 'bm25_retriever'
        elif query_class == 'semantic':
            return 'vector_retriever'
        else:
            return 'both_retrievers'

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(RetrievalState)

        workflow.add_node("query_classifier", self._classify)
        workflow.add_node("vector_retriever", self._vector_retrieve)
        workflow.add_node("bm25_retriever", self._bm25_retrieve)
        workflow.add_node("both_retrievers", self._both_retrieve)
        workflow.add_node("rrf_merger", self._merge)
        workflow.add_node("reranker", self._rerank)
        workflow.add_node("generate", self._generate)
        workflow.add_node("error_handler", self._handle_error)

        workflow.set_entry_point("query_classifier")

        workflow.add_conditional_edges(
            "query_classifier",
            self._route_decision,
            {
                "vector_retriever": "vector_retriever",
                "bm25_retriever": "bm25_retriever",
                "both_retrievers": "both_retrievers",
            }
        )

        workflow.add_edge("vector_retriever", "rrf_merger")
        workflow.add_edge("bm25_retriever", "rrf_merger")
        workflow.add_edge("both_retrievers", "rrf_merger")

        workflow.add_edge("rrf_merger", "reranker")
        workflow.add_edge("reranker", "generate")
        workflow.add_edge("error_handler", "generate")

        workflow.add_edge("generate", END)

        return workflow.compile()

    def _classify(self, state: Dict) -> Dict:
        return self.classifier.classify(state)

    def _vector_retrieve(self, state: Dict) -> Dict:
        return self.vector_retriever.retrieve(state)

    def _bm25_retrieve(self, state: Dict) -> Dict:
        return self.bm25_retriever.retrieve(state)

    def _both_retrieve(self, state: Dict) -> Dict:
        state = self.vector_retriever.retrieve(state)
        state = self.bm25_retriever.retrieve(state)
        return state

    def _merge(self, state: Dict) -> Dict:
        return self.rrf_merger.merge(state)

    def _rerank(self, state: Dict) -> Dict:
        return self.reranker.rerank(state)

    def _generate(self, state: Dict) -> Dict:
        return self.generator.generate(state)

    def _handle_error(self, state: Dict) -> Dict:
        state['answer'] = f"处理出错: {state.get('error', '未知错误')}"
        return state

    def invoke(self, query: str, **kwargs) -> Dict:
        initial_state = {
            'query': query,
            'query_class': 'balanced',
            'class_scores': {'keyword': 0.33, 'semantic': 0.33, 'balanced': 0.34},
            'query_vector': kwargs.get('query_vector'),
            'channel_weights': [0.5, 0.5],
            'retrieved_docs': [],
            'bm25_results': [],
            'merged_docs': [],
            'reranked_docs': [],
            'answer': '',
            'sources': [],
            'metadata': {},
            'error': None,
            'cached': False,
        }
        result = self.graph.invoke(initial_state)
        return result

    async def ainvoke(self, query: str, **kwargs) -> Dict:
        initial_state = {
            'query': query,
            'query_class': 'balanced',
            'class_scores': {'keyword': 0.33, 'semantic': 0.33, 'balanced': 0.34},
            'query_vector': kwargs.get('query_vector'),
            'channel_weights': [0.5, 0.5],
            'retrieved_docs': [],
            'bm25_results': [],
            'merged_docs': [],
            'reranked_docs': [],
            'answer': '',
            'sources': [],
            'metadata': {},
            'error': None,
            'cached': False,
        }
        result = await self.graph.ainvoke(initial_state)
        return result