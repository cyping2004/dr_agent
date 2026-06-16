#!/usr/bin/env python3
"""Summarize average latency by top-k and generate grouped bar charts.

This script scans ``eval/results`` for run folders named like:
``run_YYYYMMDD_HHMMSS_topkK``.

For each requested date prefix, it keeps the latest run per top-k, reads the
per-query ``metrics.csv`` file, computes the average ``fast_web`` and
``deep_rag`` latency and the deep-rag component timings, writes a table with
the concrete values, and generates a grouped bar chart.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager


RUN_FOLDER_RE = re.compile(r"^run_(\d{8})_(\d{6})_topk(\d+)$")


@dataclass(frozen=True)
class RunSummary:
    date_prefix: str
    topk: int
    run_dir: Path
    query_count: int
    fast_web_avg_ms: float
    deep_rag_avg_ms: float
    deep_rag_writer_avg_ms: float
    deep_rag_chunk_avg_ms: float
    deep_rag_embed_avg_ms: float
    deep_rag_retrieve_avg_ms: float
    deep_rag_ingest_avg_ms: float


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


def _mean_metric(rows: Sequence[Dict[str, str]], column: str) -> float:
    values: List[float] = []
    for row in rows:
        raw_value = row.get(column, "")
        if raw_value == "" or raw_value is None:
            continue
        values.append(float(raw_value))
    return mean(values) if values else 0.0


def _discover_latest_runs(results_dir: Path, date_prefix: str) -> Dict[int, Path]:
    latest_by_topk: Dict[int, tuple[int, int, Path]] = {}

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
            latest_by_topk[topk] = (timestamp, topk, path)

    return {topk: item[2] for topk, item in latest_by_topk.items()}


def _summarize_run(date_prefix: str, topk: int, run_dir: Path) -> RunSummary:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv: {metrics_path}")

    rows = _load_metrics(metrics_path)
    return RunSummary(
        date_prefix=date_prefix,
        topk=topk,
        run_dir=run_dir,
        query_count=len(rows),
        fast_web_avg_ms=_mean_metric(rows, "fast_web_total_time"),
        deep_rag_avg_ms=_mean_metric(rows, "deep_rag_total_time"),
        deep_rag_writer_avg_ms=_mean_metric(rows, "deep_rag_writer_time"),
        deep_rag_chunk_avg_ms=_mean_metric(rows, "deep_rag_chunk_time"),
        deep_rag_embed_avg_ms=_mean_metric(rows, "deep_rag_embed_time"),
        deep_rag_retrieve_avg_ms=_mean_metric(rows, "deep_rag_retrieve_time"),
        deep_rag_ingest_avg_ms=(
            _mean_metric(rows, "deep_rag_chunk_time")
            + _mean_metric(rows, "deep_rag_embed_time")
        ),
    )


def _write_summary_csv(output_path: Path, summaries: Sequence[RunSummary]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "date_prefix",
            "topk",
            "run_dir",
            "query_count",
            "fast_web_avg_ms",
            "deep_rag_avg_ms",
            "deep_rag_writer_avg_ms",
            "deep_rag_chunk_avg_ms",
            "deep_rag_embed_avg_ms",
            "deep_rag_retrieve_avg_ms",
            "deep_rag_ingest_avg_ms",
        ])
        for summary in summaries:
            writer.writerow([
                summary.date_prefix,
                summary.topk,
                summary.run_dir.name,
                summary.query_count,
                f"{summary.fast_web_avg_ms:.2f}",
                f"{summary.deep_rag_avg_ms:.2f}",
                f"{summary.deep_rag_writer_avg_ms:.2f}",
                f"{summary.deep_rag_chunk_avg_ms:.2f}",
                f"{summary.deep_rag_embed_avg_ms:.2f}",
                f"{summary.deep_rag_retrieve_avg_ms:.2f}",
                f"{summary.deep_rag_ingest_avg_ms:.2f}",
            ])


def _write_summary_markdown(output_path: Path, summaries: Sequence[RunSummary]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Latency Summary by Top-k",
        "",
        "| date_prefix | topk | run_dir | query_count | fast_web_avg_ms | deep_rag_avg_ms | deep_rag_writer_avg_ms | deep_rag_chunk_avg_ms | deep_rag_embed_avg_ms | deep_rag_retrieve_avg_ms | deep_rag_ingest_avg_ms |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.date_prefix} | {summary.topk} | {summary.run_dir.name} | "
            f"{summary.query_count} | {summary.fast_web_avg_ms:.2f} | {summary.deep_rag_avg_ms:.2f} | "
            f"{summary.deep_rag_writer_avg_ms:.2f} | {summary.deep_rag_chunk_avg_ms:.2f} | "
            f"{summary.deep_rag_embed_avg_ms:.2f} | {summary.deep_rag_retrieve_avg_ms:.2f} | "
            f"{summary.deep_rag_ingest_avg_ms:.2f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_summary(output_path: Path, summaries: Sequence[RunSummary], title: str) -> None:
    if not summaries:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = sorted(summaries, key=lambda item: item.topk)
    topks = [summary.topk for summary in summaries]
    fast_vals = [summary.fast_web_avg_ms for summary in summaries]
    deep_writer_vals = [summary.deep_rag_writer_avg_ms for summary in summaries]
    deep_ingest_vals = [summary.deep_rag_ingest_avg_ms for summary in summaries]
    deep_retrieve_vals = [summary.deep_rag_retrieve_avg_ms for summary in summaries]

    fast_avg = mean(fast_vals) if fast_vals else 0.0
    labels = ["fast_web"] + [f"topk={topk}" for topk in topks]
    x_positions = list(range(len(labels)))
    bar_width = 0.5

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fast_container = ax.bar(
        [x_positions[0]],
        [fast_avg],
        width=bar_width,
        label="fast_web",
        color="#4E79A7",
        edgecolor="white",
        linewidth=0.8,
    )

    deep_x_positions = x_positions[1:]
    writer_container = ax.bar(
        deep_x_positions,
        deep_writer_vals,
        width=bar_width,
        label="deep_rag_writer",
        color="#F28E2B",
        edgecolor="white",
        linewidth=0.8,
    )
    ingest_container = ax.bar(
        deep_x_positions,
        deep_ingest_vals,
        width=bar_width,
        bottom=deep_writer_vals,
        label="deep_rag_ingest",
        color="#E15759",
        edgecolor="white",
        linewidth=0.8,
    )
    retrieve_container = ax.bar(
        deep_x_positions,
        deep_retrieve_vals,
        width=bar_width,
        bottom=[writer + ingest for writer, ingest in zip(deep_writer_vals, deep_ingest_vals)],
        label="deep_rag_retrieve",
        color="#76B7B2",
        edgecolor="white",
        linewidth=0.8,
    )

    deep_totals = [
        writer + ingest + retrieve
        for writer, ingest, retrieve in zip(
            deep_writer_vals, deep_ingest_vals, deep_retrieve_vals
        )
    ]
    max_height = max([fast_avg] + deep_totals) if fast_avg or deep_totals else 0.0
    y_offset = max_height * 0.02 if max_height else 1.0
    side_offset = bar_width / 2 + 0.06

    if fast_avg > 0:
        ax.text(
            x_positions[0],
            fast_avg + y_offset,
            f"{fast_avg:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
        )

    for idx, x_pos in enumerate(deep_x_positions):
        cumulative = 0.0
        for value in (
            deep_writer_vals[idx],
            deep_ingest_vals[idx],
            deep_retrieve_vals[idx],
        ):
            if value <= 0:
                continue
            cumulative += value
            ax.text(
                x_pos + side_offset,
                cumulative,
                f"{value:.0f}",
                ha="left",
                va="center",
                fontsize=10,
                color="#333333",
            )

        if cumulative > 0:
            ax.text(
                x_pos,
                cumulative + y_offset,
                f"{cumulative:.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#333333",
            )

    ax.set_title("")
    ax.set_xlabel("设置")
    ax.set_ylabel("平均时延（ms）")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, color="#AFAFAF")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(
        frameon=True,
        framealpha=0.95,
        edgecolor="#D0D0D0",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.12),
    )

    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _summarize_prefix(results_dir: Path, output_dir: Path, date_prefix: str) -> List[RunSummary]:
    latest_runs = _discover_latest_runs(results_dir, date_prefix)
    summaries = [
        _summarize_run(date_prefix, topk, run_dir)
        for topk, run_dir in sorted(latest_runs.items())
    ]

    prefix_dir = output_dir / f"run_{date_prefix}"
    _write_summary_csv(prefix_dir / "latency_summary.csv", summaries)
    _write_summary_markdown(prefix_dir / "latency_summary.md", summaries)
    _plot_summary(
        prefix_dir / "latency_by_topk.svg",
        summaries,
        title="不同 Top-K 的时延",
    )

    return summaries


def main() -> None:
    _configure_plot_style()
    parser = argparse.ArgumentParser(
        description="Summarize average latency by top-k and plot grouped bars"
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
        default="eval/results/plots_latency_by_topk",
        help="Directory to write summary tables and plots",
    )
    parser.add_argument(
        "--date-prefixes",
        nargs="+",
        default=["20260429"],
        help="Date prefixes to summarize (default: 20260425 20260426)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    all_summaries: List[RunSummary] = []
    for date_prefix in args.date_prefixes:
        summaries = _summarize_prefix(results_dir, output_dir, date_prefix)
        all_summaries.extend(summaries)

    if all_summaries:
        _write_summary_csv(output_dir / "latency_summary_all.csv", all_summaries)
        _write_summary_markdown(output_dir / "latency_summary_all.md", all_summaries)

    print(f"[LatencySummary] Output dir: {output_dir}")


if __name__ == "__main__":
    main()