from rest_framework import serializers
from apps.documents.models import Collection, Document, Chunk


class ChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chunk
        fields = ['id', 'content', 'chunk_index', 'vector_id', 'metadata', 'created_at']
        read_only_fields = ['id', 'created_at']


class DocumentSerializer(serializers.ModelSerializer):
    chunks_count = serializers.SerializerMethodField()
    processing_progress = serializers.ReadOnlyField()

    class Meta:
        model = Document
        fields = [
            'id', 'collection', 'title', 'file_path', 'file_type',
            'file_size', 'status', 'error_message', 'total_chunks',
            'processed_chunks', 'processing_progress', 'chunks_count',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'status', 'error_message', 'created_at', 'updated_at']

    def get_chunks_count(self, obj):
        return obj.chunks.count()


class CollectionSerializer(serializers.ModelSerializer):
    documents_count = serializers.SerializerMethodField()
    total_chunks = serializers.SerializerMethodField()

    class Meta:
        model = Collection
        fields = [
            'id', 'name', 'description', 'is_active',
            'documents_count', 'total_chunks', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_documents_count(self, obj):
        return obj.documents.count()

    def get_total_chunks(self, obj):
        return sum(doc.chunks.count() for doc in obj.documents.all())


class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    collection_id = serializers.IntegerField()
    title = serializers.CharField(max_length=500, required=False) 