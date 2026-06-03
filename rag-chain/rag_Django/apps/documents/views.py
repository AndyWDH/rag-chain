from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404
from apps.documents.models import Collection, Document, Chunk
from apps.documents.serializers import CollectionSerializer, DocumentSerializer, ChunkSerializer
from apps.documents.services import DocumentService
import logging

logger = logging.getLogger(__name__)


class CollectionViewSet(viewsets.ModelViewSet):
    queryset = Collection.objects.all()
    serializer_class = CollectionSerializer

    def get_queryset(self):
        return Collection.objects.filter(is_active=True)


class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = (MultiPartParser, FormParser)

    def get_queryset(self):
        queryset = Document.objects.all()
        collection_id = self.request.query_params.get('collection_id')
        if collection_id:
            queryset = queryset.filter(collection_id=collection_id)
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        return queryset

    @action(detail=True, methods=['post'])
    def process(self, request, pk=None):
        document = self.get_object()
        service = DocumentService()

        try:
            result = service.process_document(document)
            return Response({
                'status': 'success',
                'message': f'Document processed, {result["chunks_created"]} chunks created',
                'chunks_created': result['chunks_created'],
            })
        except Exception as e:
            logger.error(f"Failed to process document {pk}: {e}")
            return Response({
                'status': 'error',
                'message': str(e),
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['get'])
    def chunks(self, request, pk=None):
        document = self.get_object()
        chunks = document.chunks.all()
        serializer = ChunkSerializer(chunks, many=True)
        return Response(serializer.data)


class ChunkViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Chunk.objects.all()
    serializer_class = ChunkSerializer

    def get_queryset(self):
        queryset = Chunk.objects.all()
        document_id = self.request.query_params.get('document_id')
        if document_id:
            queryset = queryset.filter(document_id=document_id)
        return queryset