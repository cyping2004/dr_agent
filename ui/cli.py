"""
CLI 界面
用于运行深度研究代理的命令行界面。
"""

import sys
from rich.console import Console
from rich.panel import Panel

from agent.state import ResearchState
from graph.research_graph import build_graph

console = Console()


def run_cli(query: str = None):
    """
    运行 CLI 界面。

    Args:
        query: 可选的初始查询。如果未提供，将提示用户输入。
    """
    console.print(
        Panel(
            "[bold cyan]🔬 Deep Research Agent - Fast Web Mode[/bold cyan]\n"
            "极速网络基线模式 - 无状态搜索",
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
        mode="fast_web",
        auto_approve=True
    )

    # 构建并运行图
    graph = build_graph()

    try:
        console.print("[dim]正在规划研究任务...[/dim]")
        result = graph.invoke(state)
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)

    # 显示结果
    console.print()
    console.print(
        Panel(
            "[bold green]✅ 研究完成[/bold green]",
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
    filename = f"report_{query.replace(' ', '_')[:30]}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(result.report_draft)

    console.print(f"[dim]报告已保存到: {filename}[/dim]")


def main():
    """主入口点。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Deep Research Agent - Fast Web Mode"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="研究查询（可选，未提供则提示输入）"
    )

    args = parser.parse_args()

    run_cli(args.query)


if __name__ == "__main__":
    main()