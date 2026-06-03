from django.urls import path, include
from rest_framework.routers import DefaultRouter
from apps.chat.views import SessionViewSet, MessageViewSet, QueryViewSet, stream_query

router = DefaultRouter()
router.register(r'sessions', SessionViewSet)
router.register(r'messages', MessageViewSet)
router.register(r'query', QueryViewSet, basename='query')

urlpatterns = [
    path('', include(router.urls)),
    path('query/stream/', stream_query, name='query-stream'),
]