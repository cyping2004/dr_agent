#!/usr/bin/env python3
"""Summarize input tokens and compression ratio by top-k.

Scans eval/results for run folders named like:
  run_YYYYMMDD_HHMMSS_topkK

For the chosen date prefix, it keeps the latest run per top-k, reads
metrics.csv, computes average fast_web/deep_rag input tokens and
compression_ratio_tokens, writes summary tables, and generates plots.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager


RUN_FOLDER_RE = re.compile(r"^run_(\d{8})_(\d{6})_topk(\d+)$")


@dataclass(frozen=True)
class RunSummary:
    date_prefix: str
    topk: int
    run_dir: Path
    total_rows: int
    kept_rows: int
    fast_web_avg_tokens: float
    deep_rag_avg_tokens: float
    compression_ratio_tokens_avg: float


def _find_cjk_font() -> Optional[Path]:
    env_path = os.getenv("CJK_FONT_PATH")
    candidates: List[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    eval_dir = Path(__file__).resolve().parents[1]
    candidates.append(eval_dir / "fonts" / "NotoSansCJKsc-Regular.otf")

    for path in candidates:
        if path.exists():
            return path
    return None


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "SimHei",
                "Microsoft YaHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "svg.fonttype": "path",
        }
    )

    font_path = _find_cjk_font()
    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
    else:
        print(
            "[图表] 未找到中文字体。可以设置环境变量 CJK_FONT_PATH，"
            "或将 NotoSansCJKsc-Regular.otf 放到 eval/fonts/ 目录。"
        )


def _load_metrics(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_float(row: Dict[str, str], column: str) -> float | None:
    raw_value = row.get(column, "")
    if raw_value in ("", None):
        return None
    try:
        return float(raw_value)
    except ValueError:
        return None


def _mean_metric(rows: Sequence[Dict[str, str]], column: str) -> float:
    values: List[float] = []
    for row in rows:
        value = _parse_float(row, column)
        if value is None:
            continue
        values.append(value)
    return mean(values) if values else 0.0


def _discover_latest_runs(results_dir: Path, date_prefix: str) -> Dict[int, Path]:
    latest_by_topk: Dict[int, tuple[int, Path]] = {}

    for path in results_dir.iterdir():
        if not path.is_dir():
            continue

        match = RUN_FOLDER_RE.match(path.name)
        if not match or match.group(1) != date_prefix:
            continue

        topk = int(match.group(3))
        timestamp = int(match.group(1) + match.group(2))
        existing = latest_by_topk.get(topk)
        if existing is None or timestamp > existing[0]:
            latest_by_topk[topk] = (timestamp, path)

    return {topk: item[1] for topk, item in latest_by_topk.items()}


def _filter_rows(rows: Sequence[Dict[str, str]], drop_zero_deep: bool) -> List[Dict[str, str]]:
    if not drop_zero_deep:
        return list(rows)

    filtered: List[Dict[str, str]] = []
    for row in rows:
        deep_value = _parse_float(row, "deep_rag_input_tokens")
        if deep_value is None or deep_value == 0:
            continue
        filtered.append(row)
    return filtered


def _summarize_run(date_prefix: str, topk: int, run_dir: Path, drop_zero_deep: bool) -> RunSummary:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv: {metrics_path}")

    rows = _load_metrics(metrics_path)
    kept_rows = _filter_rows(rows, drop_zero_deep=drop_zero_deep)

    return RunSummary(
        date_prefix=date_prefix,
        topk=topk,
        run_dir=run_dir,
        total_rows=len(rows),
        kept_rows=len(kept_rows),
        fast_web_avg_tokens=_mean_metric(rows, "fast_web_input_tokens"),
        deep_rag_avg_tokens=_mean_metric(kept_rows, "deep_rag_input_tokens"),
        compression_ratio_tokens_avg=_mean_metric(kept_rows, "compression_ratio_tokens"),
    )


def _write_summary_csv(output_path: Path, summaries: Sequence[RunSummary]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "date_prefix",
            "topk",
            "run_dir",
            "total_rows",
            "kept_rows",
            "fast_web_avg_tokens",
            "deep_rag_avg_tokens",
            "compression_ratio_tokens_avg",
        ])
        for summary in summaries:
            writer.writerow([
                summary.date_prefix,
                summary.topk,
                summary.run_dir.name,
                summary.total_rows,
                summary.kept_rows,
                f"{summary.fast_web_avg_tokens:.2f}",
                f"{summary.deep_rag_avg_tokens:.2f}",
                f"{summary.compression_ratio_tokens_avg:.2f}",
            ])


def _write_summary_markdown(output_path: Path, summaries: Sequence[RunSummary], drop_zero_deep: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drop_note = "(drop deep_rag_input_tokens==0)" if drop_zero_deep else "(keep zeros)"
    lines = [
        "# Input Tokens & Compression Ratio by Top-k",
        "",
        f"Filter: {drop_note}",
        "",
        "| date_prefix | topk | run_dir | total_rows | kept_rows | fast_web_avg_tokens | deep_rag_avg_tokens | compression_ratio_tokens_avg |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.date_prefix} | {summary.topk} | {summary.run_dir.name} | "
            f"{summary.total_rows} | {summary.kept_rows} | {summary.fast_web_avg_tokens:.2f} | "
            f"{summary.deep_rag_avg_tokens:.2f} | {summary.compression_ratio_tokens_avg:.2f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_token_bars(output_path: Path, summaries: Sequence[RunSummary], title: str) -> None:
    if not summaries:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summaries = sorted(summaries, key=lambda item: item.topk)

    topks = [summary.topk for summary in summaries]
    fast_vals = [summary.fast_web_avg_tokens for summary in summaries]
    deep_vals = [summary.deep_rag_avg_tokens for summary in summaries]

    x_positions = list(range(len(topks)))
    bar_width = 0.36
    fast_positions = [pos - bar_width / 2 for pos in x_positions]
    deep_positions = [pos + bar_width / 2 for pos in x_positions]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    fast_bars = ax.bar(
        fast_positions,
        fast_vals,
        width=bar_width,
        color="#4E79A7",
        edgecolor="white",
        linewidth=0.8,
        label="fast_web 输入 Token",
    )
    deep_bars = ax.bar(
        deep_positions,
        deep_vals,
        width=bar_width,
        color="#F28E2B",
        edgecolor="white",
        linewidth=0.8,
        label="deep_rag 输入 Token",
    )

    ax.set_title("")
    ax.set_xlabel("Top-k 值")
    ax.set_ylabel("平均输入 Token 数")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(topk) for topk in topks])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, color="#AFAFAF")
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    max_val = max(fast_vals + deep_vals) if fast_vals or deep_vals else 0
    y_offset = max_val * 0.01 if max_val else 1.0
    for bar in list(fast_bars) + list(deep_bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
        )

    ax.legend(
        frameon=True,
        framealpha=0.95,
        edgecolor="#D0D0D0",
        loc="lower left",
        bbox_to_anchor=(0.0, 1.12),
        borderaxespad=0.0,
    )
    ax.tick_params(axis="both", labelsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_compression_ratio_line(output_path: Path, summaries: Sequence[RunSummary], title: str) -> None:
    if not summaries:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summaries = sorted(summaries, key=lambda item: item.topk)

    topks = [summary.topk for summary in summaries]
    ratio_vals = [summary.compression_ratio_tokens_avg for summary in summaries]
    x_positions = list(range(len(topks)))

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(
        x_positions,
        ratio_vals,
        marker="o",
        color="#59A14F",
        linewidth=2.0,
        label="tokens压缩比",
    )

    ax.set_title(title, fontsize=18, fontweight="semibold", pad=10)
    ax.set_xlabel("Top-k 值")
    ax.set_ylabel("平均tokens压缩比")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(topk) for topk in topks])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, color="#AFAFAF")
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    max_val = max(ratio_vals) if ratio_vals else 0
    y_offset = max_val * 0.03 if max_val else 0.1
    for x_pos, value in zip(x_positions, ratio_vals):
        ax.text(
            x_pos,
            value + y_offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
        )

    ax.legend(frameon=True, framealpha=0.95, edgecolor="#D0D0D0")
    ax.tick_params(axis="both", labelsize=13)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def main() -> None:
    _configure_plot_style()
    parser = argparse.ArgumentParser(
        description="Summarize input tokens and compression ratio by top-k"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="eval/results",
        help="Directory containing run_YYYYMMDD_HHMMSS_topkK folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/results/plots_topk_tokens_compression",
        help="Directory to write summary tables and plots",
    )
    parser.add_argument(
        "--date-prefix",
        type=str,
        default="20260429",
        help="Date prefix to summarize (default: 20260429)",
    )
    parser.add_argument(
        "--keep-zero-deep",
        action="store_true",
        help="Keep rows where deep_rag_input_tokens == 0",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    date_prefix = args.date_prefix
    drop_zero_deep = not args.keep_zero_deep

    latest_runs = _discover_latest_runs(results_dir, date_prefix)
    summaries = [
        _summarize_run(date_prefix, topk, run_dir, drop_zero_deep=drop_zero_deep)
        for topk, run_dir in sorted(latest_runs.items())
    ]

    prefix_dir = output_dir / f"run_{date_prefix}"
    _write_summary_csv(prefix_dir / "topk_tokens_compression_summary.csv", summaries)
    _write_summary_markdown(prefix_dir / "topk_tokens_compression_summary.md", summaries, drop_zero_deep)
    _plot_token_bars(
        prefix_dir / "input_tokens_by_topk.svg",
        summaries,
        title="不同 Top-K 的输入 Token",
    )
    _plot_compression_ratio_line(
        prefix_dir / "compression_ratio_by_topk.svg",
        summaries,
        title="不同 Top-k 下的压缩比",
    )

    print(f"[TokensSummary] Output dir: {prefix_dir}")


if __name__ == "__main__":
    main()
