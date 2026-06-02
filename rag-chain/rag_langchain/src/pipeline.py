"""RAG Pipeline - 基于 LangChain 的高并发实现（带检索优化）"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_core.retrievers import BaseRetriever

from src.config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_K,
    RERANK_TOP_N,
    DATA_DIR,
)
from src.cache import cache_manager


class DashScopeEmbeddings(Embeddings):
    """自定义 DashScope Embedding 包装器"""
    
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


# ==================== STEP 1: Query 分类器 ====================

class QueryClassifier:
    """轻量 Query 分类器 - 基于正则、领域词典、长度和疑问句特征"""
    
    # 领域关键词词典（保险领域）
    DOMAIN_KEYWORDS = {
        "policy": ["保单", "保险", "条款", "合同", "险种", "保费", "保额", "理赔", "赔付"],
        "health": ["疾病", "医疗", "住院", "手术", "门诊", "体检", "健康", "治疗"],
        "accident": ["意外", "伤残", "身故", "烧伤", "烫伤", "骨折", "意外事故"],
        "finance": ["缴费", "缴费期", "等待期", "犹豫期", "宽限期", "现金价值", "红利"],
        "liability": ["责任", "免责", "除外", "不保", "拒绝", "不予赔付"],
    }
    
    # 疑问词模式
    QUESTION_PATTERNS = [
        r"什么是.*",
        r"什么叫.*",
        r"什么.*？",
        r"怎么.*",
        r"如何.*",
        r"为什么.*",
        r"是否.*",
        r"能不能.*",
        r"可以.*吗",
        r"应该.*",
        r"需要.*",
        r"多少.*",
        r"多久.*",
        r"哪些.*",
        r"哪些情况.*",
        r"哪些疾病.*",
        r"保.*吗",
        r"不保.*吗",
    ]
    
    # 关键词模式（偏关键词查询）
    KEYWORD_PATTERNS = [
        r"^[^\?？]+$",  # 不带问号的陈述句
        r"^[\u4e00-\u9fa5]{1,6}$",  # 短文本（1-6个汉字）
        r"^[\w]+$",  # 单个词
    ]
    
    @classmethod
    def classify(cls, query: str) -> Tuple[str, Dict[str, float]]:
        """
        分类结果：
        - "keyword": 偏关键词查询（适合 BM25）
        - "semantic": 偏语义查询（适合向量检索）
        - "balanced": 平衡型（两路都用）
        
        返回：(分类结果, 置信度字典)
        """
        features = cls._extract_features(query)
        scores = cls._calculate_scores(features)
        
        # 决策逻辑
        if scores["keyword"] > 0.6:
            return "keyword", scores
        elif scores["semantic"] > 0.6:
            return "semantic", scores
        else:
            return "balanced", scores
    
    @classmethod
    def _extract_features(cls, query: str) -> Dict[str, Any]:
        """提取查询特征"""
        features = {
            "length": len(query),
            "is_question": False,
            "question_type": None,
            "domain_keyword_count": 0,
            "domain_keyword_types": [],
            "is_short": len(query) <= 6,
            "is_long": len(query) >= 20,
            "has_special_chars": bool(re.search(r"[?？！!。，,、]", query)),
        }
        
        # 检查是否为疑问句
        for pattern in cls.QUESTION_PATTERNS:
            if re.match(pattern, query):
                features["is_question"] = True
                features["question_type"] = pattern
                break
        
        # 统计领域关键词
        for domain, keywords in cls.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    features["domain_keyword_count"] += 1
                    if domain not in features["domain_keyword_types"]:
                        features["domain_keyword_types"].append(domain)
        
        return features
    
    @classmethod
    def _calculate_scores(cls, features: Dict[str, Any]) -> Dict[str, float]:
        """计算各分类的分数"""
        keyword_score = 0.0
        semantic_score = 0.0
        
        # 长度特征
        if features["is_short"]:
            keyword_score += 0.3
        elif features["is_long"]:
            semantic_score += 0.2
        
        # 疑问句特征
        if features["is_question"]:
            semantic_score += 0.3
        else:
            keyword_score += 0.2
        
        # 领域关键词数量
        if features["domain_keyword_count"] == 1:
            keyword_score += 0.2
        elif features["domain_keyword_count"] >= 2:
            # 多个领域关键词，可能是偏语义的复杂查询
            semantic_score += 0.1
            keyword_score += 0.1
        
        # 特殊字符特征
        if not features["has_special_chars"]:
            keyword_score += 0.2
        
        # 归一化
        total = keyword_score + semantic_score + 0.1  # 加小常数避免除零
        return {
            "keyword": round(keyword_score / total, 2),
            "semantic": round(semantic_score / total, 2),
            "balanced": round(0.1 / total, 2),
        }


# ==================== STEP 2: 纯 RRF 合并器 ====================

class PureRRFRetriever(BaseRetriever):
    """纯 RRF（Reciprocal Rank Fusion）合并器 - 只看排名，不看分数"""
    
    retrievers: List = []
    weights: List[float] = []
    _k: int = 10
    
    def __init__(self, retrievers: List, weights: List[float] = None):
        """
        Args:
            retrievers: 检索器列表
            weights: 通道参与度权重（非分数权重，而是参与度）
        """
        super().__init__()
        object.__setattr__(self, 'retrievers', retrievers)
        object.__setattr__(self, 'weights', weights if weights else [1.0] * len(retrievers))
        object.__setattr__(self, '_k', 10)
    
    def _rrf_score(self, rank: int, k: int = 60) -> float:
        """计算单个文档的 RRF 分数"""
        return 1.0 / (k + rank)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """执行检索并使用 RRF 合并结果"""
        # 各通道独立检索
        all_results = []
        for idx, retriever in enumerate(self.retrievers):
            results = retriever.get_relevant_documents(query)
            # 记录排名和通道权重
            for rank, doc in enumerate(results):
                doc.metadata["_rank"] = rank + 1  # 排名从1开始
                doc.metadata["_channel_weight"] = self.weights[idx]
                doc.metadata["_channel_idx"] = idx
            all_results.extend(results)
        
        # RRF 合并
        doc_scores = {}
        for doc in all_results:
            doc_id = id(doc)
            rank = doc.metadata["_rank"]
            weight = doc.metadata["_channel_weight"]
            rrf_score = self._rrf_score(rank) * weight
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0.0}
            doc_scores[doc_id]["score"] += rrf_score
        
        # 按 RRF 分数排序
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # 返回 top k
        return [item["doc"] for item in sorted_docs[:self._k]]
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步执行检索"""
        return self._get_relevant_documents(query)
    
    def invoke(self, query: str, k: int = 10) -> List[Document]:
        """执行检索（兼容方法）"""
        self._k = k
        return self._get_relevant_documents(query)


# ==================== STEP 3: Cross-Encoder Reranker ====================

class CrossEncoderReranker:
    """Cross-Encoder 精排器"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
    
    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model
    
    def rerank(self, query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
        """对文档列表进行精排"""
        if len(docs) <= top_n:
            return docs
        
        try:
            model = self._ensure_model()
        except Exception as e:
            print(f"Cross-Encoder 加载失败，跳过精排: {e}")
            return docs[:top_n]
        
        # 准备输入对
        pairs = [(query, doc.page_content) for doc in docs]
        
        # 预测分数
        scores = model.predict(pairs)
        
        # 按分数排序
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 返回 top_n
        return [doc for doc, score in scored_docs[:top_n]]


# ==================== STEP 4: Bad Case 回归集 ====================

class BadCaseRegressor:
    """Bad Case 回归集校准器"""
    
    def __init__(self):
        self.bad_cases = []
    
    def load_bad_cases(self, bad_cases: List[Dict[str, Any]]):
        """加载 bad case 回归集"""
        self.bad_cases = bad_cases
    
    def evaluate(self, query_class: str, channel_weights: List[float]) -> float:
        """评估当前配置在回归集上的表现"""
        # 按 query 类型过滤
        filtered_cases = [
            case for case in self.bad_cases 
            if case.get("query_class") == query_class
        ]
        
        if not filtered_cases:
            return 0.0
        
        # 简单模拟评估：检查权重是否有利于该类型
        # 实际应用中应该有真实的检索评估
        score = 0.0
        for case in filtered_cases:
            expected_channel = case.get("expected_channel", "balanced")
            if expected_channel == "keyword" and channel_weights[1] > channel_weights[0]:
                score += 1
            elif expected_channel == "semantic" and channel_weights[0] > channel_weights[1]:
                score += 1
            elif expected_channel == "balanced" and abs(channel_weights[0] - channel_weights[1]) < 0.2:
                score += 1
        
        return score / len(filtered_cases)
    
    def suggest_weights(self, query_class: str, current_weights: List[float]) -> List[float]:
        """根据回归集建议调整权重"""
        filtered_cases = [
            case for case in self.bad_cases 
            if case.get("query_class") == query_class
        ]
        
        if not filtered_cases:
            return current_weights
        
        # 统计需要调整的方向
        need_more_keyword = sum(1 for c in filtered_cases if c.get("expected_channel") == "keyword")
        need_more_semantic = sum(1 for c in filtered_cases if c.get("expected_channel") == "semantic")
        
        weights = current_weights.copy()
        adjustment = 0.1
        
        if need_more_keyword > need_more_semantic:
            # 需要增强关键词通道
            weights[1] = min(1.0, weights[1] + adjustment)
            weights[0] = max(0.1, weights[0] - adjustment * 0.5)
        elif need_more_semantic > need_more_keyword:
            # 需要增强语义通道
            weights[0] = min(1.0, weights[0] + adjustment)
            weights[1] = max(0.1, weights[1] - adjustment * 0.5)
        
        return weights


# ==================== 主 RAG Pipeline ====================

class RAGPipeline:
    """LangChain 版本的 RAG 管道（带完整检索优化）"""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.llm = None
        self.reranker = None
        self.query_classifier = QueryClassifier()
        self.bad_case_regressor = BadCaseRegressor()
        
        # 通道参与度权重（可根据 query 类型动态调整）
        self.channel_weights = {
            "keyword": [0.3, 0.7],   # 偏关键词：向量0.3，BM25 0.7
            "semantic": [0.7, 0.3],  # 偏语义：向量0.7，BM25 0.3
            "balanced": [0.5, 0.5],   # 平衡：各0.5
        }
    
    def _init_llm(self):
        """初始化 LLM 客户端"""
        if self.llm is None:
            self.llm = ChatOpenAI(
                model=LLM_MODEL,
                api_key=DASHSCOPE_API_KEY,
                base_url=DASHSCOPE_BASE_URL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        return self.llm
    
    def _init_embeddings(self):
        """初始化 Embedding 模型"""
        if self.embeddings is None:
            self.embeddings = DashScopeEmbeddings(
                api_key=DASHSCOPE_API_KEY,
                base_url=DASHSCOPE_BASE_URL,
                model="text-embedding-v2"
            )
        return self.embeddings
    
    def load_and_split_documents(self, data_dir: Optional[str] = None) -> List[Document]:
        """加载并切分文档"""
        if data_dir is None:
            data_dir = DATA_DIR
        
        loader = DirectoryLoader(data_dir, loader_cls=PyPDFLoader)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        
        return splitter.split_documents(docs)
    
    def build_index(self, docs: Optional[List[Document]] = None) -> int:
        """构建向量索引"""
        if docs is None:
            docs = self.load_and_split_documents()
        
        self._init_embeddings()
        
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        
        return len(docs)
    
    def load_index(self):
        """加载已存在的向量索引"""
        self._init_embeddings()
        
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
        )
    
    def build_retriever(self):
        """构建优化后的检索器"""
        if self.vectorstore is None:
            self.load_index()
        
        # 创建基础检索器
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        
        # 获取文档用于 BM25
        all_docs = self.vectorstore.get()
        if all_docs["ids"]:
            docs = [
                Document(page_content=all_docs["documents"][i], metadata=all_docs["metadatas"][i])
                for i in range(len(all_docs["ids"]))
            ]
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = RETRIEVAL_K
            
            # 使用纯 RRF 合并器（不使用 EnsembleRetriever 的分数加权）
            self.retriever = PureRRFRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.5, 0.5]  # 默认平衡权重
            )
        else:
            self.retriever = vector_retriever
        
        return self.retriever
    
    def _init_reranker(self):
        """初始化 Cross-Encoder 精排器"""
        if self.reranker is None:
            try:
                self.reranker = CrossEncoderReranker()
            except Exception as e:
                print(f"Cross-Encoder 加载失败，将跳过精排: {e}")
                self.reranker = None
        return self.reranker
    
    def build_qa_chain(self):
        """构建问答链"""
        if self.retriever is None:
            self.build_retriever()
        
        self._init_llm()
        
        prompt_template = """你是一个保险领域的智能助手。请根据提供的上下文文档回答问题。

规则：
1. 只使用上下文中的信息回答
2. 如果上下文没有相关信息，回答"根据现有资料无法回答该问题"
3. 如果问题涉及免责条款，明确指出属于拒赔范围
4. 回答要简洁准确，引用来源文档

上下文：
{context}

问题：{question}

回答：
"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )
        
        # 使用自定义检索逻辑，不直接用 RetrievalQA 的检索器
        # 这里为了兼容性，仍然使用 RetrievalQA，但实际检索由自定义流程控制
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
        
        return self.qa_chain
    
    def query(self, query: str) -> Dict[str, Any]:
        """同步查询接口（带完整优化流程）"""
        # 检查结果缓存
        cached_result = cache_manager.get_result(query)
        if cached_result:
            return {
                "answer": cached_result,
                "sources": [],
                "cached": True,
                "query_class": "cached",
            }
        
        # STEP 1: Query 分类
        query_class, class_scores = self.query_classifier.classify(query)
        
        # 获取该类型的通道权重
        weights = self.channel_weights.get(query_class, [0.5, 0.5])
        
        # 如果有 bad case 回归集，进行权重校准
        if self.bad_case_regressor.bad_cases:
            suggested_weights = self.bad_case_regressor.suggest_weights(query_class, weights)
            # 平滑过渡到建议权重
            weights = [
                weights[i] * 0.7 + suggested_weights[i] * 0.3
                for i in range(len(weights))
            ]
        
        # 更新 RRF 检索器的权重
        if hasattr(self.retriever, 'weights'):
            self.retriever.weights = weights
        
        # STEP 2: RRF 检索
        if self.retriever is None:
            self.build_retriever()
        
        retrieved_docs = self.retriever.invoke(query, k=RETRIEVAL_K)
        
        # STEP 3: Cross-Encoder 精排
        self._init_reranker()
        if self.reranker:
            retrieved_docs = self.reranker.rerank(query, retrieved_docs, top_n=RERANK_TOP_N)
        
        # 构建上下文
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 调用 LLM 生成答案
        self._init_llm()
        prompt = PromptTemplate(
            template="""你是一个保险领域的智能助手。请根据提供的上下文文档回答问题。

规则：
1. 只使用上下文中的信息回答
2. 如果上下文没有相关信息，回答"根据现有资料无法回答该问题"
3. 如果问题涉及免责条款，明确指出属于拒赔范围
4. 回答要简洁准确，引用来源文档

上下文：
{context}

问题：{question}

回答：
""",
            input_variables=["context", "question"],
        )
        
        answer = self.llm.invoke(prompt.format(context=context, question=query)).content
        
        # 获取来源
        sources = list(set(doc.metadata.get("source", "") for doc in retrieved_docs))
        
        # 更新缓存
        cache_manager.set_result(query, answer)
        
        return {
            "answer": answer,
            "sources": sources,
            "cached": False,
            "query_class": query_class,
            "class_scores": class_scores,
            "channel_weights": weights,
        }
    
    async def aquery(self, query: str) -> Dict[str, Any]:
        """异步查询接口"""
        cached_result = await cache_manager.aget_result(query)
        if cached_result:
            return {
                "answer": cached_result,
                "sources": [],
                "cached": True,
                "query_class": "cached",
            }
        
        # 同步执行检索（简化实现）
        result = self.query(query)
        return result
    
    def count(self) -> int:
        """获取向量库中文档数量"""
        if self.vectorstore is None:
            self.load_index()
        return self.vectorstore._collection.count()
    
    def load_bad_cases(self, bad_cases: List[Dict[str, Any]]):
        """加载 bad case 回归集"""
        self.bad_case_regressor.load_bad_cases(bad_cases)
