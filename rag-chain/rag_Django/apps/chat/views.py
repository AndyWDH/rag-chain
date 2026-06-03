from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from apps.chat.models import Session, Message
from apps.chat.serializers import SessionSerializer, MessageSerializer, QuerySerializer
from apps.retrieval.service import get_retrieval_service
from django.http import StreamingHttpResponse, HttpResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
import logging
import json
import time

logger = logging.getLogger(__name__)


class SessionViewSet(viewsets.ModelViewSet):
    queryset = Session.objects.all()
    serializer_class = SessionSerializer

    def get_queryset(self):
        queryset = Session.objects.all()
        collection_id = self.request.query_params.get('collection_id')
        if collection_id:
            queryset = queryset.filter(collection_id=collection_id)
        is_active = self.request.query_params.get('is_active')
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active.lower() == 'true')
        return queryset

    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        session = self.get_object()
        messages = session.messages.all()
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)


class MessageViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Message.objects.all()
    serializer_class = MessageSerializer

    def get_queryset(self):
        queryset = Message.objects.all()
        session_id = self.request.query_params.get('session_id')
        if session_id:
            queryset = queryset.filter(session_id=session_id)
        query_class = self.request.query_params.get('query_class')
        if query_class:
            queryset = queryset.filter(query_class=query_class)
        return queryset

    @action(detail=True, methods=['post'])
    def feedback(self, request, pk=None):
        message = self.get_object()
        feedback_value = request.data.get('feedback')

        if feedback_value not in ['good', 'bad', None]:
            return Response({
                'status': 'error',
                'message': 'Invalid feedback value'
            }, status=status.HTTP_400_BAD_REQUEST)

        message.feedback = feedback_value
        message.save()

        return Response({
            'status': 'success',
            'message': f'Feedback updated to {feedback_value}'
        })


class QueryViewSet(viewsets.ViewSet):
    def create(self, request):
        serializer = QuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        query_text = serializer.validated_data['query']
        session_id = serializer.validated_data.get('session_id')
        use_cache = serializer.validated_data.get('use_cache', True)
        collection_id = serializer.validated_data.get('collection_id')

        try:
            retrieval_service = get_retrieval_service()
            result = retrieval_service.query(
                query=query_text,
                use_cache=use_cache,
                collection_id=collection_id
            )

            session = None
            if session_id:
                try:
                    session = Session.objects.get(id=session_id)
                except Session.DoesNotExist:
                    pass

            if not session and collection_id:
                session = Session.objects.create(
                    collection_id=collection_id,
                    title=query_text[:50]
                )

            if session:
                message = Message.objects.create(
                    session=session,
                    query=query_text,
                    answer=result.get('answer', ''),
                    query_class=result.get('query_class', 'balanced'),
                    class_scores=result.get('class_scores', {}),
                    channel_weights=result.get('channel_weights', [0.5, 0.5]),
                    sources=result.get('sources', []),
                    latency=result.get('latency', 0.0),
                    is_cached=result.get('cached', False),
                )
                session.update_message_count()

            return Response({
                'answer': result.get('answer', ''),
                'sources': result.get('sources', []),
                'query_class': result.get('query_class', 'balanced'),
                'class_scores': result.get('class_scores', {}),
                'channel_weights': result.get('channel_weights', [0.5, 0.5]),
                'latency': result.get('latency', 0.0),
                'cached': result.get('cached', False),
                'session_id': session.id if session else None,
            })

        except Exception as e:
            logger.error(f"Query error: {e}")
            return Response({
                'answer': f'查询出错: {str(e)}',
                'sources': [],
                'error': str(e),
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@csrf_exempt
def stream_query(request):
    """SSE 流式查询接口 - 支持 GET 和 POST 方法"""
    query_text = ''
    session_id = None
    use_cache = True
    collection_id = None

    try:
        if request.method == 'POST':
            data = json.loads(request.body.decode('utf-8'))
            query_text = data.get('query', '')
            session_id = data.get('session_id')
            use_cache = data.get('use_cache', True)
            collection_id = data.get('collection_id')
        else:
            # GET 方法用于 SSE
            query_text = request.GET.get('query', '')
            session_id = request.GET.get('session_id')
            use_cache = request.GET.get('use_cache', 'true').lower() == 'true'
            collection_id = request.GET.get('collection_id')
    except json.JSONDecodeError:
        return HttpResponse(json.dumps({'error': 'Invalid JSON'}), 
                           content_type='application/json', 
                           status=400)

    if not query_text:
        return HttpResponse(json.dumps({'error': 'Query is required'}), 
                           content_type='application/json', 
                           status=400)

    def generate_stream():
        try:
            retrieval_service = get_retrieval_service()
            result = retrieval_service.query(
                query=query_text,
                use_cache=use_cache,
                collection_id=collection_id
            )

            answer = result.get('answer', '')
            query_class = result.get('query_class', 'balanced')
            class_scores = result.get('class_scores', {})
            channel_weights = result.get('channel_weights', [0.5, 0.5])
            sources = result.get('sources', [])

            session = None
            if session_id:
                try:
                    session = Session.objects.get(id=session_id)
                except Session.DoesNotExist:
                    pass

            if not session and collection_id:
                session = Session.objects.create(
                    collection_id=collection_id,
                    title=query_text[:50]
                )

            if session:
                message = Message.objects.create(
                    session=session,
                    query=query_text,
                    answer=answer,
                    query_class=query_class,
                    class_scores=class_scores,
                    channel_weights=channel_weights,
                    sources=sources,
                    latency=result.get('latency', 0.0),
                    is_cached=result.get('cached', False),
                )
                session.update_message_count()

            metadata = {
                'query_class': query_class,
                'class_scores': class_scores,
                'channel_weights': channel_weights,
                'sources': sources
            }
            yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"

            if answer:
                chunks = [answer[i:i+50] for i in range(0, len(answer), 50)]
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        'chunk': chunk,
                        'index': i,
                        'total': len(chunks),
                        'is_last': i == len(chunks) - 1
                    }
                    yield f"event: content\ndata: {json.dumps(chunk_data)}\n\n"
                    time.sleep(0.05)
            else:
                chunk_data = {
                    'chunk': '',
                    'index': 0,
                    'total': 1,
                    'is_last': True
                }
                yield f"event: content\ndata: {json.dumps(chunk_data)}\n\n"

            finish_data = {
                'status': 'completed',
                'session_id': session.id if session else None
            }
            yield f"event: finish\ndata: {json.dumps(finish_data)}\n\n"

        except Exception as e:
            logger.error(f"Stream query error: {e}")
            error_data = {'error': str(e)}
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

    response = StreamingHttpResponse(
        generate_stream(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['Access-Control-Allow-Origin'] = '*'
    return response