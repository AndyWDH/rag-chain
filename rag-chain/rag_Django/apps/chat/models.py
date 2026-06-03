from django.db import models
from django.conf import settings
from apps.core.models import TimeStampedModel
from apps.documents.models import Collection


class Session(TimeStampedModel):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='rag_sessions',
        null=True,
        blank=True,
        verbose_name="用户"
    )
    collection = models.ForeignKey(
        Collection,
        on_delete=models.SET_NULL,
        null=True,
        related_name='sessions',
        verbose_name="关联集合"
    )
    title = models.CharField(max_length=255, blank=True, verbose_name="会话标题")
    is_active = models.BooleanField(default=True, verbose_name="是否活跃")
    message_count = models.IntegerField(default=0, verbose_name="消息数")

    class Meta:
        db_table = 'sessions'
        verbose_name = '会话'
        verbose_name_plural = '会话'
        ordering = ['-created_at']

    def __str__(self):
        return self.title or f"Session {self.id}"

    def update_message_count(self):
        self.message_count = self.messages.count()
        self.save(update_fields=['message_count'])


class Message(TimeStampedModel):
    session = models.ForeignKey(
        Session,
        on_delete=models.CASCADE,
        related_name='messages',
        verbose_name="所属会话"
    )
    query = models.TextField(verbose_name="用户查询")
    answer = models.TextField(verbose_name="系统回答")
    query_class = models.CharField(max_length=50, verbose_name="查询类型")
    class_scores = models.JSONField(default=dict, verbose_name="分类分数")
    channel_weights = models.JSONField(default=list, verbose_name="通道权重")
    sources = models.JSONField(default=list, verbose_name="来源文档")
    latency = models.FloatField(default=0.0, verbose_name="响应时间(秒)")
    is_cached = models.BooleanField(default=False, verbose_name="是否命中缓存")
    feedback = models.CharField(
        max_length=20,
        choices=[('good', '好'), ('bad', '差'), (None, '未评价')],
        null=True,
        blank=True,
        verbose_name="用户反馈"
    )

    class Meta:
        db_table = 'messages'
        verbose_name = '消息'
        verbose_name_plural = '消息'
        ordering = ['created_at']

    def __str__(self):
        return f"{self.query[:50]}..."