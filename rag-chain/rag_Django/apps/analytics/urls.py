from django.urls import path, include
from rest_framework.routers import DefaultRouter
from apps.analytics.views import BadCaseViewSet, QueryLogViewSet, RetrievalMetricsViewSet

router = DefaultRouter()
router.register(r'bad-cases', BadCaseViewSet)
router.register(r'query-logs', QueryLogViewSet)
router.register(r'metrics', RetrievalMetricsViewSet)

urlpatterns = [
    path('', include(router.urls)),
]