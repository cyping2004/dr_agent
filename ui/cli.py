"""
CLI 界面
用于运行深度研究代理的命令行界面。
"""

import hashlib
import re
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from agent.state import ResearchState
from graph.research_graph import build_graph

console = Console()


def run_cli(
    query: str | None = None,
    mode: str = "fast_web",
    test_mode: bool = False,
    output_path: str | None = None,
):
    """
    运行 CLI 界面。

    Args:
        query: 可选的初始查询。如果未提供，将提示用户输入。
    """
    mode_label = "Fast Web" if mode == "fast_web" else "Deep RAG"
    test_label = " + Test Mode" if test_mode else ""
    console.print(
        Panel(
            f"[bold cyan]🔬 Deep Research Agent - {mode_label}{test_label}[/bold cyan]\n"
            "CLI 运行模式",
            border_style="cyan"
        )
    )
    console.print()

    # 获取查询
    if query is None:
        query = console.input("[bold yellow]请输入研究查询:[/bold yellow] ").strip()

    if not query:
        console.print("[red]错误: 查询不能为空[/red]")
        sys.exit(1)

    console.print(f"\n[bold]正在处理查询:[/bold] {query}\n")

    # 创建初始状态
    state = ResearchState(
        query=query,
        mode=mode,
        test_mode=test_mode,
        hitl_enabled=False,
    )

    # 构建并运行图
    graph = build_graph()

    try:
        console.print("[dim]正在规划研究任务...[/dim]")
        result = graph.invoke(state)
        if isinstance(result, dict):
            result = ResearchState(**result)
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)

    # 显示结果
    console.print()
    console.print(
        Panel(
            "[bold green]研究完成[/bold green]",
            border_style="green"
        )
    )
    console.print()

    console.print("[bold]研究任务:[/bold]")
    for i, task in enumerate(result.research_tasks, 1):
        console.print(f"  {i}. {task}")
    console.print()

    console.print("[bold]研究报告:[/bold]")
    console.print(result.report_draft)
    console.print()

    # 保存报告到文件
    output_file = _resolve_output_path(query, output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(result.report_draft, encoding="utf-8")

    console.print(f"[dim]报告已保存到: {output_file}[/dim]")


def _resolve_output_path(query: str, output_path: str | None) -> Path:
    """Resolve report output path from user input."""
    default_name = _default_report_name(query)

    if not output_path:
        return Path("reports") / default_name

    path = Path(output_path)
    if path.exists() and path.is_dir():
        return path / default_name

    if str(output_path).endswith(":") or str(output_path).endswith("/"):
        path.mkdir(parents=True, exist_ok=True)
        return path / default_name

    return path


def _default_report_name(query: str) -> str:
    """Build a safe default filename for the report."""
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", query.strip())
    safe = safe.strip("_")
    if not safe:
        safe = f"{_short_hash(query)}"
    return f"report_{safe[:30]}.md"


def _short_hash(text: str) -> str:
    """Generate a short, stable ASCII hash for non-ASCII queries."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def main():
    """主入口点。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Deep Research Agent CLI"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="研究查询（可选，未提供则提示输入）"
    )
    parser.add_argument(
        "--mode",
        choices=["fast_web", "deep_rag"],
        default="fast_web",
        help="运行模式: fast_web | deep_rag"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="启用测试模式（复用 Web 搜索结果）"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="报告保存路径（可为文件或目录）"
    )

    args = parser.parse_args()

    run_cli(args.query, mode=args.mode, test_mode=args.test_mode, output_path=args.output)


if __name__ == "__main__":
    main()