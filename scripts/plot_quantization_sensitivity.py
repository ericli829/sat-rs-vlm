"""
量化敏感度可视化脚本

生成量化敏感度测试结果的可视化图表。

用法：
    python scripts/plot_quantization_sensitivity.py \
        --input reports/quantization_sensitivity \
        --output reports/quantization_sensitivity/figures
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np


def load_sensitivity_report(report_dir: str) -> dict:
    """加载敏感度报告"""
    report_file = os.path.join(report_dir, "sensitivity_report.json")
    if not os.path.exists(report_file):
        raise FileNotFoundError(f"报告文件不存在: {report_file}")

    with open(report_file, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_component_sensitivity(report: dict, output_dir: str):
    """绘制组件敏感度对比图"""
    layer_results = report.get("layer_results", [])

    if not layer_results:
        print("没有找到层结果数据")
        return

    # 准备数据
    names = [r["layer_name"] for r in layer_results]
    sensitivities = [r["sensitivity_score"] for r in layer_results]
    keyword_deltas = [r["keyword_hit_delta"] * 100 for r in layer_results]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 敏感度对比
    colors = ['green' if s < 0.02 else 'orange' if s < 0.1 else 'red' for s in sensitivities]
    bars = ax1.bar(range(len(names)), sensitivities, color=colors)
    ax1.set_xlabel('组件')
    ax1.set_ylabel('敏感度评分')
    ax1.set_title('各组件量化敏感度对比')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.axhline(y=0.02, color='green', linestyle='--', alpha=0.5, label='低敏感阈值')
    ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='高敏感阈值')
    ax1.legend()

    # Keyword Hit 变化对比
    colors2 = ['green' if d >= 0 else 'red' for d in keyword_deltas]
    bars2 = ax2.bar(range(len(names)), keyword_deltas, color=colors2)
    ax2.set_xlabel('组件')
    ax2.set_ylabel('Keyword Hit Rate 变化 (%)')
    ax2.set_title('量化后 Keyword Hit Rate 变化')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "component_sensitivity.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_dir}/component_sensitivity.png")


def plot_sensitivity_distribution(report: dict, output_dir: str):
    """绘制敏感度分布图"""
    layer_results = report.get("layer_results", [])

    if not layer_results:
        return

    sensitivities = [r["sensitivity_score"] for r in layer_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    # 直方图
    ax.hist(sensitivities, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('敏感度评分')
    ax.set_ylabel('频次')
    ax.set_title('量化敏感度分布')
    ax.axvline(x=0.02, color='green', linestyle='--', alpha=0.5, label='低敏感阈值')
    ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='高敏感阈值')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sensitivity_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_dir}/sensitivity_distribution.png")


def plot_latency_comparison(report: dict, output_dir: str):
    """绘制延迟对比图"""
    layer_results = report.get("layer_results", [])

    if not layer_results:
        return

    names = [r["layer_name"] for r in layer_results]
    baseline_latency = [r["baseline_latency_ms"] for r in layer_results]
    quantized_latency = [r["quantized_latency_ms"] for r in layer_results]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_latency, width, label='量化前 (FP32)', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, quantized_latency, width, label='量化后 (INT8)', color='orange', alpha=0.7)

    ax.set_xlabel('组件')
    ax.set_ylabel('推理延迟 (ms)')
    ax.set_title('量化前后推理延迟对比')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_dir}/latency_comparison.png")


def plot_tradeoff(report: dict, output_dir: str):
    """绘制精度-速度 trade-off 图"""
    layer_results = report.get("layer_results", [])

    if not layer_results:
        return

    sensitivities = [r["sensitivity_score"] for r in layer_results]
    latency_changes = [r["latency_change_ratio"] * 100 for r in layer_results]
    names = [r["layer_name"] for r in layer_results]

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(latency_changes, sensitivities, s=100, c=sensitivities,
                        cmap='RdYlGn_r', edgecolors='black', alpha=0.7)

    # 添加标签
    for i, name in enumerate(names):
        ax.annotate(name, (latency_changes[i], sensitivities[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_xlabel('延迟变化 (%)')
    ax.set_ylabel('敏感度评分')
    ax.set_title('精度-速度 Trade-off 分析')
    ax.axhline(y=0.02, color='green', linestyle='--', alpha=0.5, label='低敏感阈值')
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='高敏感阈值')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.legend()

    plt.colorbar(scatter, label='敏感度')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tradeoff_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_dir}/tradeoff_analysis.png")


def main():
    parser = argparse.ArgumentParser(description="量化敏感度可视化")
    parser.add_argument("--input", required=True, help="输入报告目录")
    parser.add_argument("--output", required=True, help="输出图表目录")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 加载报告
    report = load_sensitivity_report(args.input)

    # 生成图表
    print("生成可视化图表...")
    plot_component_sensitivity(report, args.output)
    plot_sensitivity_distribution(report, args.output)
    plot_latency_comparison(report, args.output)
    plot_tradeoff(report, args.output)

    print(f"\n所有图表已保存到: {args.output}")


if __name__ == "__main__":
    main()
