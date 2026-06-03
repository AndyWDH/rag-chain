from django.urls import path
from apps.retrieval.views import HealthCheckView, StatsView

urlpatterns = [
    path('health/', HealthCheckView.as_view(), name='health'),
    path('stats/', StatsView.as_view(), name='stats'),
]