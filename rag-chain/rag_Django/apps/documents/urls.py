from django.urls import path, include
from rest_framework.routers import DefaultRouter
from apps.documents.views import CollectionViewSet, DocumentViewSet, ChunkViewSet

router = DefaultRouter()
router.register(r'collections', CollectionViewSet)
router.register(r'documents', DocumentViewSet)
router.register(r'chunks', ChunkViewSet)

urlpatterns = [
    path('', include(router.urls)),
]