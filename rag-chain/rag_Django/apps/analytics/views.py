from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.db.models import Count, Avg, F
from django.utils import timezone
from datetime import timedelta
from apps.analytics.models import BadCase, QueryLog, RetrievalMetrics
from apps.analytics.serializers import BadCaseSerializer, QueryLogSerializer, RetrievalMetricsSerializer
from apps.retrieval.service import get_retrieval_service
from apps.documents.models import Collection
import logging

logger = logging.getLogger(__name__)


class BadCaseViewSet(viewsets.ModelViewSet):
    queryset = BadCase.objects.all()
    serializer_class = BadCaseSerializer

    def get_queryset(self):
        queryset = BadCase.objects.all()
        collection_id = self.request.query_params.get('collection_id')
        if collection_id:
            queryset = queryset.filter(collection_id=collection_id)
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        query_class = self.request.query_params.get('query_class')
        if query_class:
            queryset = queryset.filter(query_class=query_class)
        return queryset

    @action(detail=True, methods=['post'])
    def resolve(self, request, pk=None):
        bad_case = self.get_object()
        resolution_notes = request.data.get('resolution_notes', '')

        bad_case.status = 'resolved'
        bad_case.resolution_notes = resolution_notes
        bad_case.save()

        return Response({
            'status': 'success',
            'message': 'Bad case marked as resolved'
        })

    @action(detail=False, methods=['post'])
    def test_weights(self, request):
        query = request.data.get('query')
        query_class = request.data.get('query_class', 'balanced')

        if not query:
            return Response({
                'status': 'error',
                'message': 'Query is required'
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            retrieval_service = get_retrieval_service()
            result = retrieval_service.query(query)

            return Response({
                'status': 'success',
                'query': query,
                'query_class': result.get('query_class'),
                'answer': result.get('answer'),
                'channel_weights': result.get('channel_weights'),
            })
        except Exception as e:
            logger.error(f"Weight test error: {e}")
            return Response({
                'status': 'error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class QueryLogViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = QueryLog.objects.all()
    serializer_class = QueryLogSerializer

    def get_queryset(self):
        queryset = QueryLog.objects.all()
        query_class = self.request.query_params.get('query_class')
        if query_class:
            queryset = queryset.filter(query_class=query_class)
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')
        if start_date:
            queryset = queryset.filter(created_at__gte=start_date)
        if end_date:
            queryset = queryset.filter(created_at__lte=end_date)
        return queryset


class RetrievalMetricsViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = RetrievalMetrics.objects.all()
    serializer_class = RetrievalMetricsSerializer

    @action(detail=False, methods=['get'])
    def summary(self, request):
        today = timezone.now().date()
        week_ago = today - timedelta(days=7)

        recent_metrics = RetrievalMetrics.objects.filter(date__gte=week_ago)

        if not recent_metrics.exists():
            return Response({
                'total_queries': 0,
                'avg_response_time': 0,
                'cache_hit_rate': 0,
            })

        summary = recent_metrics.aggregate(
            total_queries=Count('id'),
            avg_response_time=Avg('avg_response_time'),
            cache_hit_rate=Avg('cache_hit_rate'),
        )

        return Response(summary)

    @action(detail=False, methods=['post'])
    def refresh(self, request):
        today = timezone.now().date()
        yesterday = today - timedelta(days=1)

        recent_logs = QueryLog.objects.filter(
            created_at__date=yesterday
        )

        if not recent_logs.exists():
            return Response({
                'status': 'success',
                'message': 'No logs to process'
            })

        metrics_data = {
            'date': yesterday,
            'total_queries': recent_logs.count(),
            'avg_response_time': recent_logs.aggregate(
                avg=Avg('response_time')
            )['avg'] or 0,
            'cache_hit_rate': recent_logs.filter(
                cache_hit=True
            ).count() / recent_logs.count() * 100,
            'keyword_queries': recent_logs.filter(
                query_class='keyword'
            ).count(),
            'semantic_queries': recent_logs.filter(
                query_class='semantic'
            ).count(),
            'balanced_queries': recent_logs.filter(
                query_class='balanced'
            ).count(),
        }

        metrics, created = RetrievalMetrics.objects.update_or_create(
            date=yesterday,
            defaults=metrics_data
        )

        return Response({
            'status': 'success',
            'metrics_id': metrics.id,
            'created': created,
        })