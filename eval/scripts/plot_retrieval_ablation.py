#!/usr/bin/env python3
"""从 CSV 结果生成检索消融图（仅召回率，柱状图）。"""

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def _find_cjk_font() -> Optional[Path]:
    env_path = os.getenv("CJK_FONT_PATH")
    candidates = []
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


METRICS = ("recall",)

# 随机检索基线的 Recall@k（8.33%、13.89%、27.78%、55.56%）。
RANDOM_BASELINE_RECALL = {
    3: 0.0833,
    5: 0.1389,
    10: 0.2778,
    20: 0.5556,
}

# 适合论文图的色盲友好配色。
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

    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    data_by_label: Dict[str, Dict[int, Dict[str, float]]] = {}
    for label, csv_path in items:
        rows = _load_csv(csv_path)
        data_by_label[label] = _aggregate(rows)

    # 特殊处理：多模态视图去重
    if group_name == "多模态视图":
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
            label="随机基线",
            color=BASELINE_COLOR,
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_facecolor("#FAFAFA")
    ax.set_title(f"{group_name} Recall@k", fontsize=18, fontweight="semibold", pad=10)
    ax.set_xlabel("k 值")
    ax.set_ylabel("召回率")
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
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(fontsize=12, frameon=True, framealpha=0.95, edgecolor="#D0D0D0")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def main() -> None:
    _configure_plot_style()
    parser = argparse.ArgumentParser(description="绘制检索消融结果（仅召回率）")
    parser.add_argument(
        "--k-tag",
        type=str,
        default="k3_5_10_20",
        help="结果目录名中的 k 列表标签（例如：k3_5_10_20）",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="eval/results",
        help="包含消融结果的目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出图表目录",
    )
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir or f"eval/results/plots_{args.k_tag}")

    groups = [
        ("不同检索模式下", f"abl_{args.k_tag}_mode_", "retrieval_mode.svg"),
        ("多模态视图", f"abl_{args.k_tag}_view_", "multimodal_view.svg"),
        ("切块大小", f"abl_{args.k_tag}_chunk_", "chunk_size.svg"),
    ]

    for title, prefix, filename in groups:
        items = _collect_group(base_dir, prefix)
        _plot_group(title, items, out_dir / filename)

    print(f"[图表] 输出目录：{out_dir}")


if __name__ == "__main__":
    main()