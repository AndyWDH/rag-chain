from rest_framework import serializers
from apps.chat.models import Session, Message


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = [
            'id', 'session', 'query', 'answer', 'query_class',
            'class_scores', 'channel_weights', 'sources', 'latency',
            'is_cached', 'feedback', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class SessionSerializer(serializers.ModelSerializer):
    messages_count = serializers.ReadOnlyField()

    class Meta:
        model = Session
        fields = [
            'id', 'user', 'collection', 'title', 'is_active',
            'message_count', 'messages_count', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class QuerySerializer(serializers.Serializer):
    query = serializers.CharField(max_length=5000)
    session_id = serializers.IntegerField(required=False, allow_null=True)
    collection_id = serializers.IntegerField(required=False, allow_null=True)
    use_cache = serializers.BooleanField(default=True)


class QueryResponseSerializer(serializers.Serializer):
    answer = serializers.CharField()
    sources = serializers.ListField(child=serializers.CharField())
    query_class = serializers.CharField()
    class_scores = serializers.DictField()
    channel_weights = serializers.ListField()
    latency = serializers.FloatField()
    cached = serializers.BooleanField()
    session_id = serializers.IntegerField(allow_null=True)