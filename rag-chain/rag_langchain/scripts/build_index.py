"""一键建库脚本 - 使用 LangChain 构建向量索引"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from src.pipeline import RAGPipeline
from src.config import CHROMA_PERSIST_DIR, DATA_DIR, validate_config

console = Console()


def main() -> None:
    if not any(Path(DATA_DIR).glob("*.pdf")):
        console.print("[red]ERROR: data/sample_docs/ 下没有 PDF。[/red]")
        console.print("先生成测试数据: python scripts/generate_synthetic_data.py")
        sys.exit(1)
    
    validate_config()
    
    console.print("Building RAG pipeline...")
    pipe = RAGPipeline()
    
    console.print(f"Loading documents from {DATA_DIR}...")
    docs = pipe.load_and_split_documents()
    console.print(f"Loaded {len(docs)} chunks")
    
    console.print("Building index...")
    n = pipe.build_index(docs)
    total = pipe.count()
    
    console.print(f"\nIndex built successfully:")
    console.print(f"- Written: {n} chunks")
    console.print(f"- Total in store: {total} chunks")
    console.print(f"- Vector store: {CHROMA_PERSIST_DIR}")
    
    console.print('\nNext: python scripts/ask.py "等待期是多少天?"')


if __name__ == "__main__":
    main()
