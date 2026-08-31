#!/usr/bin/env python3
"""Generate presentation-ready Region Retriever figures from checked-in reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "evaluation"
OUTPUT_DIR = ROOT / "docs" / "architecture" / "figures" / "region_retriever"
ANALYSIS = REPORT_DIR / "rs_clip_deep_analysis_fixed_vrsbench200.json"
GATE = REPORT_DIR / "remoteclip_gate_calibration_vrsbench200.json"
OVERLAY = REPORT_DIR / "remoteclip_locator_smoke_overlay.png"
COLD = REPORT_DIR / "remoteclip_cache_cold20.json"
HOT = REPORT_DIR / "remoteclip_cache_hot20.json"

FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
if FONT_PATH.is_file():
    font_manager.fontManager.addfont(str(FONT_PATH))
    FONT_NAME = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
else:
    FONT_NAME = "DejaVu Sans"

plt.rcParams.update(
    {
        "font.family": FONT_NAME,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titleweight": "bold",
        "axes.edgecolor": "#D5DCE5",
        "axes.labelcolor": "#263342",
        "xtick.color": "#52616F",
        "ytick.color": "#52616F",
        "text.color": "#182230",
    }
)

NAVY = "#175A7E"
BLUE = "#2684C2"
GREEN = "#2B8A66"
GOLD = "#D99A2B"
RED = "#C94C4C"
PURPLE = "#7667B1"
GRAY = "#8A96A3"
LIGHT = "#E8EEF3"
DARK = "#182230"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canvas(title: str, subtitle: str = "") -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.80, bottom=0.14)
    fig.suptitle(title, x=0.06, y=0.97, ha="left", fontsize=24, fontweight="bold")
    if subtitle:
        fig.text(0.06, 0.875, subtitle, ha="left", fontsize=11, color="#52616F")
    return fig, ax


def footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.96, 0.035, text, ha="right", fontsize=8.5, color="#6B7785")


def save(fig: plt.Figure, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return path


def rounded_box(ax, xy, width, height, text, *, color=LIGHT, edge=NAVY, text_color=DARK):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=color,
        edgecolor=edge,
        linewidth=1.6,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        color=text_color,
        linespacing=1.35,
    )


def arrow(ax, start, end, *, color=GRAY, style="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color=color,
            linestyle=style,
        )
    )


def system_positioning() -> Path:
    fig, ax = canvas(
        "Region Retriever 在整套系统中的位置",
        "模型只负责给候选区域排序；TaskGraph 仍然保持模型无关",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    xs = [0.01, 0.18, 0.35, 0.52, 0.70, 0.87]
    labels = [
        "UHR 问题",
        "Text Planner\nTaskGraph",
        "LOCATE\nCapability",
        "RemoteCLIP\n3x3 区域评分",
        "Top-5\n全局 ROI",
        "LAE / Qwen\n后续处理",
    ]
    colors = ["#F4F6F8", "#E8EEF3", "#E8EEF3", "#DCEFE7", "#E8EEF3", "#F4F6F8"]
    edges = [GRAY, NAVY, NAVY, GREEN, NAVY, GRAY]
    for x, label, color, edge in zip(xs, labels, colors, edges, strict=True):
        rounded_box(ax, (x, 0.56), 0.12, 0.20, label, color=color, edge=edge)
    for left, right in zip(xs[:-1], xs[1:], strict=True):
        arrow(ax, (left + 0.122, 0.66), (right - 0.008, 0.66))
    rounded_box(
        ax,
        (0.20, 0.13),
        0.18,
        0.16,
        "VRSBench-200\n离线评测数据",
        color="#F7F1E3",
        edge=GOLD,
    )
    rounded_box(
        ax,
        (0.45, 0.13),
        0.18,
        0.16,
        "公平 benchmark\n五模型 + 统计检验",
        color="#F7F1E3",
        edge=GOLD,
    )
    rounded_box(
        ax,
        (0.70, 0.13),
        0.18,
        0.16,
        "模型定版与配置\nRemoteCLIP 默认",
        color="#DCEFE7",
        edge=GREEN,
    )
    arrow(ax, (0.38, 0.21), (0.44, 0.21), color=GOLD)
    arrow(ax, (0.63, 0.21), (0.69, 0.21), color=GOLD)
    arrow(ax, (0.79, 0.30), (0.59, 0.55), color=GREEN, style="--")
    ax.text(0.01, 0.86, "生产主链", fontsize=12, color=NAVY, fontweight="bold")
    ax.text(0.01, 0.34, "离线验证链", fontsize=12, color=GOLD, fontweight="bold")
    ax.text(
        0.50,
        0.01,
        "关键边界：VRSBench 只用于离线评测，不改变 01 中的生产数据路由",
        ha="center",
        fontsize=10.5,
        color=RED,
    )
    return save(fig, "01_system_positioning.png")


def model_ranking(analysis: dict) -> Path:
    fig, ax = canvas(
        "五个 RS-CLIP 的公平排名",
        "同一 corrected VRSBench-200、同一 3x3 candidates、Top-5、类别 query、CPU",
    )
    models = analysis["models"]
    labels = [item["model"].replace("-ViT-B-32", "").replace("1_ViT-B-32", "") for item in models]
    r1 = [100 * item["metrics"]["recall_at_1"] for item in models]
    r3 = [100 * item["metrics"]["recall_at_3"] for item in models]
    r5 = [100 * item["metrics"]["recall_at_5"] for item in models]
    x = np.arange(len(labels))
    width = 0.23
    bars1 = ax.bar(x - width, r1, width, label="Recall@1", color=GRAY)
    bars3 = ax.bar(x, r3, width, label="Recall@3", color=BLUE)
    bars5 = ax.bar(x + width, r5, width, label="Recall@5", color=GREEN)
    ax.axhline(39.4, color=GOLD, linestyle="--", linewidth=1.6, label="随机 Top-5 39.4%")
    ax.axhline(71.0, color=RED, linestyle=":", linewidth=1.8, label="3x3 Oracle 71.0%")
    ax.set_ylabel("召回率（%）")
    ax.set_xlabel("模型")
    ax.set_ylim(0, 80)
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color="#E6EBF0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False)
    for bars in (bars1, bars3, bars5):
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.annotate(
        "当前默认",
        xy=(0 + width, r5[0]),
        xytext=(0.65, 75),
        arrowprops={"arrowstyle": "->", "color": GREEN},
        color=GREEN,
        fontsize=11,
        fontweight="bold",
    )
    footer(fig, "数据：rs_clip_deep_analysis_fixed_vrsbench200.json")
    return save(fig, "02_model_ranking.png")


def latency_tradeoff(analysis: dict) -> Path:
    fig, ax = canvas(
        "效果与 CPU 延迟的权衡",
        "RemoteCLIP、GeoRSCLIP、FarSLIP 性能接近；Git-RSCLIP 明显更慢且更弱",
    )
    latency = {
        "RemoteCLIP-ViT-B-32": 648,
        "GeoRSCLIP-ViT-B-32": 653,
        "FarSLIP1_ViT-B-32": 655,
        "SatelliteCLIP": 707,
        "Git-RSCLIP-base": 13088,
    }
    colors = [GREEN, BLUE, GOLD, PURPLE, RED]
    for item, color in zip(analysis["models"], colors, strict=True):
        name = item["model"]
        x = latency[name]
        y = 100 * item["metrics"]["recall_at_5"]
        size = 280 if name.startswith("Remote") else 180
        ax.scatter(x, y, s=size, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        short = name.replace("-ViT-B-32", "").replace("1_ViT-B-32", "")
        if name.startswith("Remote"):
            offset = (8, 14)
        elif name.startswith("Geo"):
            offset = (10, -3)
        elif name.startswith("Far"):
            offset = (10, -20)
        else:
            offset = (8, -14)
        ax.annotate(short, (x, y), xytext=offset, textcoords="offset points", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(500, 18000)
    ax.set_ylim(40, 68)
    ax.set_xlabel("单图稳态 CPU 中位延迟（ms，对数轴）")
    ax.set_ylabel("Recall@5（%）")
    ax.grid(color="#E6EBF0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvspan(580, 800, color=GREEN, alpha=0.07)
    ax.text(610, 42, "前三名均在约 0.65 秒", color=GREEN, fontsize=10)
    footer(fig, "延迟：隔离进程，warm-up 3 + measured 20")
    return save(fig, "03_quality_latency_tradeoff.png")


def target_size(analysis: dict) -> Path:
    fig, ax = canvas(
        "不同目标尺寸下的 Recall@5",
        "large target 的主要问题是单格候选覆盖不足，而不是 CLIP 排序能力",
    )
    keys = ["tiny(<0.5%)", "small(0.5-2%)", "medium(2-10%)", "large(>=10%)"]
    labels = ["Tiny\n<0.5%", "Small\n0.5-2%", "Medium\n2-10%", "Large\n>=10%"]
    selected = analysis["models"][:3]
    first = selected[0]
    series = [
        ("RemoteCLIP", [100 * first["by_size"][key]["recall_at_5"] for key in keys], GREEN),
        (
            "GeoRSCLIP",
            [100 * selected[1]["by_size"][key]["recall_at_5"] for key in keys],
            BLUE,
        ),
        (
            "FarSLIP",
            [100 * selected[2]["by_size"][key]["recall_at_5"] for key in keys],
            GOLD,
        ),
        ("随机", [100 * first["by_size"][key]["random_recall_at_k"] for key in keys], GRAY),
        ("Oracle", [100 * first["by_size"][key]["oracle_recall"] for key in keys], RED),
    ]
    x = np.arange(len(keys))
    width = 0.15
    for index, (name, values, color) in enumerate(series):
        offset = (index - 2) * width
        bars = ax.bar(x + offset, values, width, label=name, color=color)
        if name in {"RemoteCLIP", "Oracle"}:
            ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 110)
    ax.set_xlabel("目标面积占整图比例")
    ax.set_ylabel("Recall@5（%）")
    ax.grid(axis="y", color="#E6EBF0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.09), frameon=False)
    ax.annotate(
        "Oracle 也只有 30%",
        xy=(3 + 2 * width, 30),
        xytext=(2.45, 68),
        arrowprops={"arrowstyle": "->", "color": RED},
        color=RED,
        fontsize=11,
        fontweight="bold",
    )
    footer(fig, "样本数：Tiny 39 / Small 44 / Medium 47 / Large 70")
    return save(fig, "04_target_size_recall.png")


def cache_speedup(cold: dict, hot: dict) -> Path:
    fig, ax = canvas(
        "RemoteCLIP 缓存收益",
        "同一 20 条、180 tiles；热缓存排序结果完全不变",
    )
    cold_ms = cold["metrics"]["latency_ms"]
    hot_ms = hot["metrics"]["latency_ms"]
    values = [cold_ms, hot_ms]
    bars = ax.bar([0, 1], values, width=0.48, color=[GRAY, GREEN])
    ax.set_xticks([0, 1], ["首次运行\n36/180 hits", "热缓存\n180/180 hits"])
    ax.set_ylabel("平均延迟（ms / sample）")
    ax.set_xlabel("缓存状态")
    ax.set_ylim(0, cold_ms * 1.23)
    ax.grid(axis="y", color="#E6EBF0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars, labels=[f"{cold_ms:.1f} ms", f"{hot_ms:.1f} ms"], padding=6, fontsize=12)
    speedup = cold_ms / hot_ms
    ax.annotate(
        f"约 {speedup:.1f}x",
        xy=(1, hot_ms),
        xytext=(0.52, cold_ms * 0.72),
        arrowprops={"arrowstyle": "->", "color": GREEN, "linewidth": 2},
        color=GREEN,
        fontsize=22,
        fontweight="bold",
        ha="center",
    )
    ax.text(
        0.5,
        cold_ms * 0.40,
        "decoded image  |  image embedding  |  query embedding  |  disk score",
        ha="center",
        fontsize=11,
        color=NAVY,
    )
    footer(fig, "注意：首次运行包含模型装载与同批次重复图像带来的部分命中")
    return save(fig, "05_cache_speedup.png")


def count_gate(gate: dict) -> Path:
    fig, ax = canvas(
        "Count gate：高召回与调用减少的权衡",
        "目标是尽量不漏 positive tile；收益不是 Top-K 排名，而是安全排除负区域",
    )
    points = gate["in_sample_tradeoff"]
    reduction = [100 * item["detector_call_reduction"] for item in points]
    recall = [100 * item["gate_recall"] for item in points]
    ax.plot(reduction, recall, color=BLUE, linewidth=2.4, marker="o", markersize=8)
    for item, x, y in zip(points, reduction, recall, strict=True):
        ax.annotate(
            f"阈值 {item['threshold']:.3f}",
            (x, y),
            xytext=(6, -16 if x < 10 else 8),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.scatter(
        100 * gate["test"]["detector_call_reduction"],
        100 * gate["test"]["gate_recall"],
        s=260,
        color=GREEN,
        edgecolor="white",
        linewidth=1.5,
        zorder=4,
        label="留出集：100% recall / 6.63% reduction",
    )
    ax.axhline(99, color=RED, linestyle="--", linewidth=1.5, label="目标 GateRecall 99%")
    ax.set_xlim(0, 23)
    ax.set_ylim(93.5, 100.7)
    ax.set_xlabel("Detector 调用减少（%）")
    ax.set_ylabel("Positive-tile GateRecall（%）")
    ax.grid(color="#E6EBF0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False)
    ax.text(
        13.2,
        97.4,
        "继续追求调用减少会快速牺牲召回\n当前 gate 达标，但加速收益有限",
        color=RED,
        fontsize=11,
    )
    footer(fig, "校准/留出：按图像 SHA-256 分组，91 / 109 rows")
    return save(fig, "06_count_gate_tradeoff.png")


def bottleneck() -> Path:
    fig, ax = canvas(
        "当前真正的瓶颈：候选生成上限",
        "RemoteCLIP 已取得 Oracle 可用提升的约 77.8%，下一步应改候选，而不是纠结前三名",
    )
    labels = ["随机 Top-5", "RemoteCLIP", "3x3 Oracle", "理想系统"]
    values = [39.4, 64.0, 71.0, 100.0]
    colors = [GRAY, GREEN, RED, LIGHT]
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.56)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Recall@5 上限 / 实际值（%）")
    ax.set_ylabel("阶段")
    ax.grid(axis="x", color="#E6EBF0", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=5, fontsize=12)
    ax.annotate(
        "排序差距仅 7 pp",
        xy=(67.5, 1.5),
        xytext=(52, 2.45),
        arrowprops={"arrowstyle": "<->", "color": GOLD, "linewidth": 2},
        color=GOLD,
        fontsize=11,
        fontweight="bold",
    )
    ax.annotate(
        "候选生成仍缺 29 pp",
        xy=(85.5, 2.5),
        xytext=(74, 3.35),
        arrowprops={"arrowstyle": "<->", "color": RED, "linewidth": 2},
        color=RED,
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.025,
        "Top-5 同时保留约 55.6% 图像面积：当前还是 coarse retrieval baseline",
        fontsize=11,
        color=NAVY,
        transform=ax.transAxes,
    )
    footer(fig, "结论：优先测试 overlapping / multi-cell / multiscale candidates")
    return save(fig, "07_candidate_bottleneck.png")


def locator_overlay() -> Path:
    fig = plt.figure(figsize=(12, 6.75), dpi=160, facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.04)
    left = fig.add_subplot(grid[0, 0])
    right = fig.add_subplot(grid[0, 1])
    image = Image.open(OVERLAY).convert("RGB")
    left.imshow(image)
    left.axis("off")
    left.set_title("真实 RemoteCLIP 定位结果", loc="left", fontsize=20, fontweight="bold", pad=14)
    right.axis("off")
    right.set_xlim(0, 1)
    right.set_ylim(0, 1)
    right.text(0.05, 0.91, "查询", fontsize=11, color="#52616F")
    right.text(0.05, 0.84, "Where is the windmill?", fontsize=17, fontweight="bold")
    facts = [
        ("Provider", "RemoteCLIP ViT-B/32"),
        ("一级候选", "3x3 = 9 tiles"),
        ("最终输出", "5 global ROIs"),
        ("坐标", "absolute original-image xyxy"),
        ("附带信息", "score + model + tile/level provenance"),
    ]
    top = 0.69
    for index, (label, value) in enumerate(facts):
        y = top - index * 0.115
        right.text(0.05, y, label, fontsize=10, color="#6B7785")
        right.text(0.32, y, value, fontsize=11.5, color=DARK)
        right.plot([0.05, 0.95], [y - 0.04, y - 0.04], color="#E6EBF0", linewidth=1)
    right.plot([0.08, 0.15], [0.12, 0.12], color=RED, linewidth=4)
    right.text(0.18, 0.12, "红框：被选中的 core", va="center", fontsize=10.5)
    right.plot([0.55, 0.62], [0.12, 0.12], color=GOLD, linewidth=3)
    right.text(0.65, 0.12, "黄框：带 halo 的 view", va="center", fontsize=10.5)
    fig.text(
        0.965,
        0.025,
        "artifact: remoteclip_locator_smoke_overlay.png",
        ha="right",
        fontsize=8.5,
        color="#6B7785",
    )
    return save(fig, "08_real_locator_overlay.png")


def delivery_checklist() -> Path:
    fig, ax = canvas(
        "02 计划书的八项最终交付",
        "接口、真实模型、评测和可视化已经形成闭环",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    items = [
        ("01", "RegionRetriever\ninterface"),
        ("02", "Fake provider"),
        ("03", "RemoteCLIP\n真实 baseline"),
        ("04", "Benchmark\n与深度统计"),
        ("05", "四级 cache\n与 batch encode"),
        ("06", "Overlay\n与 ROI crops"),
        ("07", "GPU 生产配置\n参数可替换"),
        ("08", "结果表\n与模型推荐"),
    ]
    for index, (number, label) in enumerate(items):
        row, col = divmod(index, 4)
        x = 0.015 + col * 0.245
        y = 0.55 - row * 0.38
        rounded_box(ax, (x, y), 0.215, 0.25, label, color="#E6F2EC", edge=GREEN)
        ax.text(x + 0.02, y + 0.20, number, fontsize=10, color=GREEN, fontweight="bold")
        ax.text(x + 0.195, y + 0.205, "DONE", ha="right", fontsize=8.5, color=GREEN)
    ax.text(
        0.5,
        0.03,
        "工程完成不等于效果封顶：下一阶段仍需目标域、GPU 全链路与候选生成实验",
        ha="center",
        fontsize=11,
        color=RED,
    )
    return save(fig, "09_delivery_checklist.png")


def roadmap() -> Path:
    fig, ax = canvas(
        "下一阶段实验优先级",
        "先解决决定上限的问题，再补目标域与部署证据",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    items = [
        ("P0", "候选生成", "overlap / multi-cell / multiscale", RED),
        ("P0", "目标域验证", "MME RealWorld RS / XLRS", RED),
        ("P1", "GPU 全链路", "RemoteCLIP + LAE + Qwen\n延迟 / VRAM / OOM", GOLD),
        ("P1", "Query 与跨家族", "原句 / 类别 / 属性关系\nVisRAG / SigLIP", BLUE),
    ]
    xs = [0.02, 0.27, 0.52, 0.77]
    for index, ((priority, title, detail, color), x) in enumerate(zip(items, xs, strict=True)):
        rounded_box(ax, (x, 0.34), 0.20, 0.36, f"{title}\n\n{detail}", color="#F4F6F8", edge=color)
        ax.text(x + 0.02, 0.64, priority, fontsize=10, color=color, fontweight="bold")
        if index < len(xs) - 1:
            arrow(ax, (x + 0.205, 0.52), (xs[index + 1] - 0.008, 0.52), color=GRAY)
    ax.text(
        0.5,
        0.18,
        "判断标准：最终 VQA 正确率提升 + detector 调用减少 + GPU 端到端吞吐",
        ha="center",
        fontsize=12,
        color=NAVY,
    )
    return save(fig, "10_next_experiments.png")


def contact_sheet(paths: list[Path]) -> Path:
    fig, axes = plt.subplots(2, 5, figsize=(15, 7.8), dpi=160, facecolor="white")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04, wspace=0.06, hspace=0.16)
    fig.suptitle("Region Retriever 汇报图集", x=0.02, ha="left", fontsize=24, fontweight="bold")
    titles = [
        "系统位置",
        "模型排名",
        "效果/延迟",
        "尺寸表现",
        "缓存收益",
        "Count gate",
        "候选瓶颈",
        "真实 overlay",
        "八项交付",
        "下一步",
    ]
    for ax, path, title in zip(axes.flat, paths, titles, strict=True):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.axis("off")
        ax.set_title(title, fontsize=11, pad=5)
    return save(fig, "00_presentation_overview.png")


def main() -> int:
    analysis = load_json(ANALYSIS)
    gate = load_json(GATE)
    cold = load_json(COLD)
    hot = load_json(HOT)
    paths = [
        system_positioning(),
        model_ranking(analysis),
        latency_tradeoff(analysis),
        target_size(analysis),
        cache_speedup(cold, hot),
        count_gate(gate),
        bottleneck(),
        locator_overlay(),
        delivery_checklist(),
        roadmap(),
    ]
    contact_sheet(paths)
    print(f"generated {len(paths) + 1} figures in {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
