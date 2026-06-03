from rest_framework import serializers
from apps.analytics.models import BadCase, QueryLog, RetrievalMetrics


class BadCaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = BadCase
        fields = [
            'id', 'collection', 'query', 'expected_answer', 'actual_answer',
            'query_class', 'suggested_weights', 'current_weights',
            'status', 'resolution_notes', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class QueryLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = QueryLog
        fields = [
            'id', 'query', 'query_class', 'response_time',
            'cache_hit', 'result_count', 'error', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class RetrievalMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = RetrievalMetrics
        fields = [
            'id', 'date', 'total_queries', 'avg_response_time',
            'cache_hit_rate', 'keyword_queries', 'semantic_queries',
            'balanced_queries', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']