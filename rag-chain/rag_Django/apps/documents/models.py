from django.db import models
from apps.core.models import TimeStampedModel, StatusModel


class Collection(TimeStampedModel):
    name = models.CharField(max_length=255, unique=True, verbose_name="集合名称")
    description = models.TextField(blank=True, verbose_name="集合描述")
    is_active = models.BooleanField(default=True, verbose_name="是否启用")

    class Meta:
        db_table = 'collections'
        verbose_name = '文档集合'
        verbose_name_plural = '文档集合'
        ordering = ['-created_at']

    def __str__(self):
        return self.name


class Document(TimeStampedModel, StatusModel):
    collection = models.ForeignKey(
        Collection,
        on_delete=models.CASCADE,
        related_name='documents',
        verbose_name="所属集合"
    )
    title = models.CharField(max_length=500, verbose_name="文档标题")
    file_path = models.CharField(max_length=1000, verbose_name="文件路径")
    file_type = models.CharField(max_length=50, verbose_name="文件类型")
    file_size = models.IntegerField(default=0, verbose_name="文件大小(字节)")
    total_chunks = models.IntegerField(default=0, verbose_name="总块数")
    processed_chunks = models.IntegerField(default=0, verbose_name="已处理块数")

    class Meta:
        db_table = 'documents'
        verbose_name = '文档'
        verbose_name_plural = '文档'
        ordering = ['-created_at']

    def __str__(self):
        return self.title

    @property
    def processing_progress(self):
        if self.total_chunks == 0:
            return 0
        return int(self.processed_chunks / self.total_chunks * 100)


class Chunk(TimeStampedModel):
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='chunks',
        verbose_name="所属文档"
    )
    content = models.TextField(verbose_name="内容")
    chunk_index = models.IntegerField(verbose_name="块索引")
    vector_id = models.CharField(max_length=255, blank=True, null=True, verbose_name="向量ID")
    metadata = models.JSONField(default=dict, verbose_name="元数据")

    class Meta:
        db_table = 'chunks'
        verbose_name = '文档块'
        verbose_name_plural = '文档块'
        ordering = ['chunk_index']
        indexes = [
            models.Index(fields=['document', 'chunk_index']),
        ]

    def __str__(self):
        return f"{self.document.title} - Chunk {self.chunk_index}"