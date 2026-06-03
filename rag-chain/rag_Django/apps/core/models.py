from django.db import models
from django.utils import timezone


class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        abstract = True


class StatusModel(models.Model):
    STATUS_CHOICES = [
        ('pending', '待处理'),
        ('processing', '处理中'),
        ('completed', '已完成'),
        ('failed', '失败'),
    ]

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name="状态")
    error_message = models.TextField(blank=True, null=True, verbose_name="错误信息")

    class Meta:
        abstract = True


class QueryTypeModel(models.Model):
    QUERY_CLASS_CHOICES = [
        ('keyword', '关键词'),
        ('semantic', '语义'),
        ('balanced', '平衡'),
    ]

    class Meta:
        abstract = True