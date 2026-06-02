"""简化测试 - 验证 LLM 连接"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI
from src.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL, validate_config

def main():
    try:
        validate_config()
        
        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            temperature=0.3,
        )
        
        response = llm.invoke("你好，介绍一下自己")
        print("LLM 响应:", response.content)
        return True
    except Exception as e:
        print(f"错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
