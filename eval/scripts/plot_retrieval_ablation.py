#!/usr/bin/env python3
"""Generate retrieval ablation plots (Recall-only, bar charts) from CSV results."""

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


METRICS = ("recall",)

# Random retrieval baseline recall@k (8.33%, 13.89%, 27.78%, 55.56%).
RANDOM_BASELINE_RECALL = {
    3: 0.0833,
    5: 0.1389,
    10: 0.2778,
    20: 0.5556,
}

# Colorblind-friendly palette suitable for paper figures.
PAPER_PALETTE = [
    "#4E79A7",
    "#59A14F",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#B07AA1",
]
BASELINE_COLOR = "#9E9E9E"


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _aggregate(rows: List[Dict[str, str]]) -> Dict[int, Dict[str, float]]:
    buckets: Dict[int, Dict[str, float]] = {}
    counts: Dict[int, int] = {}

    for row in rows:
        k = int(row["k"])
        buckets.setdefault(k, {m: 0.0 for m in METRICS})
        counts[k] = counts.get(k, 0) + 1
        for metric in METRICS:
            buckets[k][metric] += float(row[metric])

    for k, totals in buckets.items():
        count = counts.get(k, 1)
        for metric in METRICS:
            totals[metric] = totals[metric] / count

    return buckets


def _strip_suffix(label: str) -> str:
    return re.sub(r"_(\d{8}(_\d{6})?)$", "", label)


def _extract_timestamp(name: str) -> int:
    match = re.search(r"_(\d{8})(?:_(\d{6}))?$", name)
    if not match:
        return -1
    date = match.group(1)
    time = match.group(2) or "000000"
    return int(f"{date}{time}")


def _series_equal(
    left: Dict[int, Dict[str, float]],
    right: Dict[int, Dict[str, float]],
    tol: float = 1e-9,
) -> bool:
    if left.keys() != right.keys():
        return False
    for k in left:
        for metric in METRICS:
            if abs(left[k][metric] - right[k][metric]) > tol:
                return False
    return True


def _collect_group(base_dir: Path, prefix: str) -> List[Tuple[str, Path]]:
    latest_by_label: Dict[str, Tuple[int, Path]] = {}
    for name in sorted(os.listdir(base_dir)):
        if not name.startswith(prefix):
            continue
        path = base_dir / name / "retrieval_metrics.csv"
        if not path.exists():
            continue
        label = _strip_suffix(name[len(prefix):])
        timestamp = _extract_timestamp(name)
        existing = latest_by_label.get(label)
        if existing is None or timestamp > existing[0]:
            latest_by_label[label] = (timestamp, path)

    return [(label, data[1]) for label, data in sorted(latest_by_label.items())]


def _plot_group(group_name: str, items: List[Tuple[str, Path]], output_path: Path) -> None:
    if not items:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    data_by_label: Dict[str, Dict[int, Dict[str, float]]] = {}
    for label, csv_path in items:
        rows = _load_csv(csv_path)
        data_by_label[label] = _aggregate(rows)

    # 特殊处理：Multimodal View 去重
    if group_name == "Multimodal View":
        caption_data = data_by_label.get("caption")
        image_data = data_by_label.get("image")
        if caption_data and image_data and _series_equal(caption_data, image_data):
            items = [(label, path) for label, path in items if label != "caption"]
            data_by_label.pop("caption", None)

    labels = [label for label, _ in items]
    ks = sorted(next(iter(data_by_label.values())).keys())

    x = np.arange(len(ks))

    baseline_vals = np.array([RANDOM_BASELINE_RECALL.get(k, np.nan) for k in ks], dtype=float)
    has_baseline = not np.all(np.isnan(baseline_vals))
    series_count = len(labels) + (1 if has_baseline else 0)
    width = 0.8 / series_count if series_count else 0.8

    for i, label in enumerate(labels):
        data = data_by_label[label]
        ys = [data[k]["recall"] for k in ks]
        bar_idx = i + (1 if has_baseline else 0)
        offset = (bar_idx - series_count / 2) * width + width / 2
        ax.bar(
            x + offset,
            ys,
            width,
            label=label,
            color=PAPER_PALETTE[i % len(PAPER_PALETTE)],
            edgecolor="white",
            linewidth=0.8,
        )

    if has_baseline:
        baseline_idx = 0
        baseline_offset = (baseline_idx - series_count / 2) * width + width / 2
        valid_mask = ~np.isnan(baseline_vals)
        ax.bar(
            x[valid_mask] + baseline_offset,
            baseline_vals[valid_mask],
            width,
            label="random-baseline",
            color=BASELINE_COLOR,
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_facecolor("#FAFAFA")
    ax.set_title(f"{group_name} (Recall@k)", fontsize=12, fontweight="semibold", pad=10)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.set_ylim(0.0, 1.0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, color="#AFAFAF")
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=8, frameon=True, framealpha=0.95, edgecolor="#D0D0D0")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot retrieval ablation results (Recall only)")
    parser.add_argument(
        "--k-tag",
        type=str,
        default="k3_5_10_20",
        help="k-list tag used in result folder names (e.g., k3_5_10_20)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="eval/results",
        help="Directory containing ablation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write plots",
    )
    parser.add_argument(
        "--include-chunk",
        action="store_true",
        help="Include chunk-size ablation group",
    )
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir or f"eval/results/plots_{args.k_tag}")

    groups = [
        ("Retrieval Mode", f"abl_{args.k_tag}_mode_", "retrieval_mode.png"),
        ("Multimodal View", f"abl_{args.k_tag}_view_", "multimodal_view.png"),
    ]
    if args.include_chunk:
        groups.append(("Chunk Size", f"abl_{args.k_tag}_chunk_", "chunk_size.png"))

    for title, prefix, filename in groups:
        items = _collect_group(base_dir, prefix)
        _plot_group(title, items, out_dir / filename)

    print(f"[Plots] Output dir: {out_dir}")


if __name__ == "__main__":
    main()