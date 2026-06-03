from django.db import models
from apps.core.models import TimeStampedModel
from apps.documents.models import Collection


class BadCase(TimeStampedModel):
    STATUS_CHOICES = [
        ('active', '活跃'),
        ('resolved', '已解决'),
        ('ignored', '已忽略'),
    ]

    collection = models.ForeignKey(
        Collection,
        on_delete=models.CASCADE,
        null=True,
        related_name='bad_cases',
        verbose_name="关联集合"
    )
    query = models.TextField(verbose_name="查询内容")
    expected_answer = models.TextField(verbose_name="期望回答")
    actual_answer = models.TextField(blank=True, verbose_name="实际回答")
    query_class = models.CharField(max_length=50, verbose_name="查询类型")
    suggested_weights = models.JSONField(default=list, verbose_name="建议权重")
    current_weights = models.JSONField(default=list, verbose_name="当前权重")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active', verbose_name="状态")
    resolution_notes = models.TextField(blank=True, verbose_name="解决备注")

    class Meta:
        db_table = 'bad_cases'
        verbose_name = 'Bad Case'
        verbose_name_plural = 'Bad Cases'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.query[:50]}... ({self.query_class})"


class QueryLog(TimeStampedModel):
    query = models.TextField(verbose_name="查询内容")
    query_class = models.CharField(max_length=50, verbose_name="查询类型")
    response_time = models.FloatField(verbose_name="响应时间(秒)")
    cache_hit = models.BooleanField(default=False, verbose_name="缓存命中")
    result_count = models.IntegerField(default=0, verbose_name="结果数量")
    error = models.TextField(blank=True, verbose_name="错误信息")

    class Meta:
        db_table = 'query_logs'
        verbose_name = '查询日志'
        verbose_name_plural = '查询日志'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['query_class', 'created_at']),
            models.Index(fields=['-response_time']),
        ]

    def __str__(self):
        return f"{self.query[:50]}... ({self.response_time}s)"


class RetrievalMetrics(TimeStampedModel):
    date = models.DateField(verbose_name="日期")
    total_queries = models.IntegerField(default=0, verbose_name="总查询数")
    avg_response_time = models.FloatField(default=0.0, verbose_name="平均响应时间")
    cache_hit_rate = models.FloatField(default=0.0, verbose_name="缓存命中率")
    keyword_queries = models.IntegerField(default=0, verbose_name="关键词查询数")
    semantic_queries = models.IntegerField(default=0, verbose_name="语义查询数")
    balanced_queries = models.IntegerField(default=0, verbose_name="平衡查询数")

    class Meta:
        db_table = 'retrieval_metrics'
        verbose_name = '检索指标'
        verbose_name_plural = '检索指标'
        ordering = ['-date']
        unique_together = ['date']

    def __str__(self):
        return f"Metrics {self.date}"