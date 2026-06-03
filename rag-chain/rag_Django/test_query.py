import requests
import json

print("=== 测试 SSE 流式查询接口 ===")
print(f"服务地址: http://localhost:8000")
print()

query = input("请输入查询问题: ")

if not query:
    query = "什么是保险？"

print(f"\n正在查询: {query}")
print("-" * 50)

try:
    response = requests.post(
        "http://localhost:8000/api/v1/chat/query/stream/",
        json={"query": query},
        stream=True
    )
    response.raise_for_status()
    
    print("\n📡 流式响应:")
    full_answer = ""
    
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            print(decoded_line)
            
            # 提取内容
            if decoded_line.startswith('data: '):
                try:
                    data = json.loads(decoded_line[5:])
                    if 'chunk' in data:
                        full_answer += data['chunk']
                except:
                    pass
    
    print("\n" + "=" * 50)
    print(f"完整回答: {full_answer}")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")