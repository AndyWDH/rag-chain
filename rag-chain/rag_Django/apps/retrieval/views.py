from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from apps.documents.vectorstore import get_vector_store_manager
from apps.documents.bm25 import get_bm25_manager
from apps.retrieval.service import get_retrieval_service
from django.conf import settings


class HealthCheckView(APIView):
    def get(self, request):
        vector_manager = get_vector_store_manager()
        bm25_manager = get_bm25_manager()

        return Response({
            'status': 'healthy',
            'services': {
                'vector_store': 'connected' if vector_manager._vector_store else 'disconnected',
                'bm25_index': 'connected' if bm25_manager._bm25_index else 'disconnected',
                'retrieval_graph': 'ready',
            },
            'settings': {
                'retrieval_k': settings.RETRIEVAL_K,
                'rerank_top_n': settings.RERANK_TOP_N,
                'rrf_k': settings.RRF_K,
            }
        })


class StatsView(APIView):
    def get(self, request):
        vector_manager = get_vector_store_manager()
        bm25_manager = get_bm25_manager()

        return Response({
            'vector_store': {
                'document_count': vector_manager.count(),
            },
            'bm25_index': {
                'document_count': bm25_manager.count(),
            },
            'channel_weights': settings.CHANNEL_WEIGHTS,
            'cache_ttl': settings.CACHE_TTL,
        })