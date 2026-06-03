import sys
import os

sys.path.insert(0, '.')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()

from apps.retrieval.service import get_retrieval_service

print("=== 最终验证测试 ===")
print()

service = get_retrieval_service()

queries = [
    "犹豫期几天",
    "什么是保险",
    "理赔需要什么材料"
]

for query in queries:
    print(f"查询: {query}")
    result = service.query(query, use_cache=False)
    print(f"  分类: {result.get('query_class')}")
    print(f"  答案: {result.get('answer')}")
    print()