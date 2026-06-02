"""问答脚本 - 使用 LangChain 进行 RAG 问答"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from src.pipeline import RAGPipeline
from src.config import validate_config

console = Console()


def main() -> None:
    if len(sys.argv) < 2:
        console.print("[red]ERROR: 请提供问题作为参数[/red]")
        console.print("Usage: python scripts/ask.py \"你的问题\"")
        sys.exit(1)
    
    validate_config()
    
    query = sys.argv[1]
    
    console.print("Loading pipeline...")
    pipe = RAGPipeline()
    pipe.build_qa_chain()
    
    console.print(f"Query: {query}")
    console.print("Processing...")
    
    result = pipe.query(query)
    
    # 显示分类信息
    if "query_class" in result and not result["cached"]:
        query_class = result["query_class"]
        class_scores = result.get("class_scores", {})
        channel_weights = result.get("channel_weights", [0.5, 0.5])
        
        console.print(f"\n[blue]Query Classification:[/blue] {query_class}")
        console.print(f"[blue]Classification Scores:[/blue] {class_scores}")
        console.print(f"[blue]Channel Weights (vector:bm25):[/blue] {channel_weights[0]:.2f}:{channel_weights[1]:.2f}")
    
    console.print(f"\n[bold green]Answer:[/bold green]")
    console.print(result["answer"])
    
    if result["sources"]:
        console.print(f"\n[dim]Sources: {', '.join(result['sources'])}[/dim]")
    
    if result["cached"]:
        console.print("[dim](Result from cache)[/dim]")


if __name__ == "__main__":
    main()