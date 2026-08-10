"""
层量化敏感度测试脚本

逐层测试模型在 INT8 量化后的精度变化，生成敏感度分析报告。
用于识别对量化最敏感的层，指导混合精度量化策略。

用法：
    python scripts/quantization_sensitivity_test.py \
        --model-dir D:\Models\Qwen3-VL-2B-Instruct \
        --data-dir D:\project\database\VRSBench\metadata \
        --image-dir D:\project\database\VRSBench\Images\Images_val \
        --output-dir reports/quantization_sensitivity \
        --num-samples 20 \
        --method layer_wise
"""

import argparse
import json
import os
import sys
import time
import copy
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Any

import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass
class LayerSensitivityResult:
    """单层量化敏感度结果"""
    layer_name: str
    layer_type: str
    num_params: int = 0
    # 量化前指标
    baseline_keyword_hit: float = 0.0
    baseline_exact_match: float = 0.0
    baseline_latency_ms: float = 0.0
    # 量化后指标
    quantized_keyword_hit: float = 0.0
    quantized_exact_match: float = 0.0
    quantized_latency_ms: float = 0.0
    # 变化
    keyword_hit_delta: float = 0.0
    exact_match_delta: float = 0.0
    latency_change_ratio: float = 0.0
    # 敏感度评分 (0-1, 越高越敏感)
    sensitivity_score: float = 0.0


@dataclass
class SensitivityTestReport:
    """敏感度测试报告"""
    model_name: str
    test_date: str
    num_samples: int
    method: str
    # 基线结果
    baseline_keyword_hit: float = 0.0
    baseline_exact_match: float = 0.0
    baseline_latency_ms: float = 0.0
    # 全量量化结果
    full_quantized_keyword_hit: float = 0.0
    full_quantized_exact_match: float = 0.0
    full_quantized_latency_ms: float = 0.0
    # 各层结果
    layer_results: list = field(default_factory=list)
    # 敏感层排名
    sensitive_layers: list = field(default_factory=list)
    # 建议
    recommendations: list = field(default_factory=list)


def load_vrsbench_data(data_dir: str, task_type: str = "vqa") -> list:
    """加载 VRSBench 数据"""
    with open(os.path.join(data_dir, f"VRSBench_EVAL_{task_type}.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_model(model, processor, val_data: list, image_dir: str,
                   num_samples: int = None) -> dict:
    """评估模型"""
    from PIL import Image

    # 如果 num_samples 为 None 或 0，使用全部数据
    if num_samples is None or num_samples <= 0:
        eval_data = val_data
    else:
        eval_data = val_data[:num_samples]

    predictions = []
    for item in eval_data:
        try:
            image_id = item.get("image_id")
            if not image_id:
                continue

            image_path = os.path.join(image_dir, image_id)
            if not os.path.exists(image_path):
                continue

            question = item.get("question", "")
            reference = item.get("ground_truth", "")

            if not question:
                continue

            image = Image.open(image_path).convert("RGB")
            msg = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}]
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt")

            start_time = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=128)
            latency = (time.perf_counter() - start_time) * 1000

            pred_text = processor.decode(outputs[0], skip_special_tokens=True)
            pred_answer = pred_text.split(question)[-1].strip() if question in pred_text else pred_text.strip()

            predictions.append({
                "prediction": pred_answer,
                "reference": reference,
                "latency_ms": latency,
            })

        except Exception as e:
            continue

    if not predictions:
        return {"keyword_hit_rate": 0.0, "exact_match_rate": 0.0, "latency_ms": 0.0}

    # 计算指标
    import re
    def keyword_hit(pred, ref):
        tokens = set(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", ref.lower()))
        if not tokens:
            return False
        return any(token in pred.lower() for token in tokens)

    keyword_hits = sum(keyword_hit(p["prediction"], p["reference"]) for p in predictions)
    exact_matches = sum(
        p["prediction"].strip().lower() == p["reference"].strip().lower()
        for p in predictions
    )

    return {
        "keyword_hit_rate": keyword_hits / len(predictions),
        "exact_match_rate": exact_matches / len(predictions),
        "latency_ms": np.mean([p["latency_ms"] for p in predictions]),
        "num_samples": len(predictions),
    }


def get_linear_layers(model) -> dict:
    """获取模型中所有 Linear 层及其分组"""
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers[name] = module
    return layers


def group_layers_by_component(layers: dict) -> dict:
    """按组件分组"""
    groups = {
        "visual_encoder": [],
        "cross_modal": [],
        "language_model": [],
        "other": [],
    }

    for name in layers:
        if "visual" in name or "vision" in name:
            groups["visual_encoder"].append(name)
        elif "cross_attn" in name or "connector" in name or "projector" in name:
            groups["cross_modal"].append(name)
        elif "language_model" in name or "model.layers" in name:
            groups["language_model"].append(name)
        else:
            groups["other"].append(name)

    return groups


def quantize_layer(model: nn.Module, layer_name: str) -> nn.Module:
    """量化指定层"""
    import torch.quantization as quantization

    # 创建模型副本
    model_copy = copy.deepcopy(model)

    # 找到要量化的层
    parts = layer_name.split(".")
    module = model_copy
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)

    # 量化该层
    if isinstance(module, nn.Linear):
        quantized = quantization.quantize_dynamic(
            module,
            {nn.Linear},
            dtype=torch.qint8,
        )
        # 替换原层
        parent = model_copy
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        setattr(parent, parts[-1], quantized)

    return model_copy


def calculate_sensitivity_score(baseline: dict, quantized: dict) -> float:
    """计算敏感度评分 (0-1, 越高越敏感)"""
    keyword_hit_drop = baseline["keyword_hit_rate"] - quantized["keyword_hit_rate"]
    latency_increase = (quantized["latency_ms"] - baseline["latency_ms"]) / max(baseline["latency_ms"], 1)

    # 敏感度 = 精度下降 * 0.7 + 延迟变化 * 0.3
    score = max(0, keyword_hit_drop) * 0.7 + max(0, latency_increase) * 0.3
    return min(1.0, score)


def generate_recommendations(results: list) -> list:
    """根据敏感度结果生成建议"""
    recommendations = []

    # 按敏感度排序
    sorted_results = sorted(results, key=lambda x: x.sensitivity_score, reverse=True)

    # 找出高敏感层
    sensitive_layers = [r for r in sorted_results if r.sensitivity_score > 0.1]
    if sensitive_layers:
        recommendations.append(
            f"发现 {len(sensitive_layers)} 个高敏感层，建议保持 FP32 精度"
        )

    # 找出低敏感层
    insensitive_layers = [r for r in sorted_results if r.sensitivity_score < 0.02]
    if insensitive_layers:
        recommendations.append(
            f"发现 {len(insensitive_layers)} 个低敏感层，可安全量化为 INT8"
        )

    # 按组件给出建议
    visual_sensitive = [r for r in sensitive_layers if "visual" in r.layer_name]
    if visual_sensitive:
        recommendations.append(
            f"视觉编码器包含 {len(visual_sensitive)} 个敏感层，建议保持 FP16/FP32"
        )

    lm_sensitive = [r for r in sensitive_layers if "language_model" in r.layer_name or "model.layers" in r.layer_name]
    if lm_sensitive:
        recommendations.append(
            f"语言模型包含 {len(lm_sensitive)} 个敏感层，建议使用混合精度量化"
        )

    return recommendations


def run_layer_wise_test(args) -> SensitivityTestReport:
    """逐层量化敏感度测试 (按层组测试)"""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    import torch.quantization as quantization

    print("=" * 60)
    print("层量化敏感度测试 (按层组)")
    print("=" * 60)

    # 加载验证数据
    val_data = load_vrsbench_data(args.data_dir)
    print(f"验证集样本数: {len(val_data)}")

    # 加载基座模型
    print("\n加载基座模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

    # 评估基线
    print("\n评估基线模型...")
    baseline_metrics = evaluate_model(model, processor, val_data, args.image_dir, args.num_samples)
    print(f"  Keyword Hit Rate: {baseline_metrics['keyword_hit_rate']:.2%}")
    print(f"  Latency: {baseline_metrics['latency_ms']:.0f}ms")

    # 获取所有 Linear 层
    linear_layers = get_linear_layers(model)
    print(f"\n找到 {len(linear_layers)} 个 Linear 层")

    # 按层组分组 (每 6 层为一组，跳过视觉编码器)
    lm_layers = [(name, module) for name, module in linear_layers.items() if "visual" not in name]
    layer_groups = {}
    group_size = 6
    for i in range(0, len(lm_layers), group_size):
        group_name = f"language_model_layers_{i//group_size + 1}"
        layer_groups[group_name] = [name for name, _ in lm_layers[i:i+group_size]]

    print(f"分为 {len(layer_groups)} 组进行测试")

    # 逐组测试
    layer_results = []

    for group_name, layer_names in layer_groups.items():
        print(f"\n测试组: {group_name} ({len(layer_names)} 层)")

        try:
            # 创建模型副本
            model_copy = copy.deepcopy(model)

            # 量化该组的所有层
            for layer_name in layer_names:
                parts = layer_name.split(".")
                module = model_copy
                for part in parts:
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)

                if isinstance(module, nn.Linear):
                    quantized = quantization.quantize_dynamic(
                        module, {nn.Linear}, torch.qint8
                    )
                    # 替换原层
                    parent = model_copy
                    for part in parts[:-1]:
                        if part.isdigit():
                            parent = parent[int(part)]
                        else:
                            parent = getattr(parent, part)
                    setattr(parent, parts[-1], quantized)

            model_copy.eval()

            # 评估
            quantized_metrics = evaluate_model(model_copy, processor, val_data, args.image_dir, args.num_samples)

            # 计算敏感度
            sensitivity = calculate_sensitivity_score(baseline_metrics, quantized_metrics)

            # 统计参数量
            num_params = 0
            for layer_name in layer_names:
                parts = layer_name.split(".")
                module = model
                for part in parts:
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)
                num_params += sum(p.numel() for p in module.parameters())

            result = LayerSensitivityResult(
                layer_name=group_name,
                layer_type="LayerGroup",
                num_params=num_params,
                baseline_keyword_hit=baseline_metrics["keyword_hit_rate"],
                baseline_exact_match=baseline_metrics["exact_match_rate"],
                baseline_latency_ms=baseline_metrics["latency_ms"],
                quantized_keyword_hit=quantized_metrics["keyword_hit_rate"],
                quantized_exact_match=quantized_metrics["exact_match_rate"],
                quantized_latency_ms=quantized_metrics["latency_ms"],
                keyword_hit_delta=quantized_metrics["keyword_hit_rate"] - baseline_metrics["keyword_hit_rate"],
                exact_match_delta=quantized_metrics["exact_match_rate"] - baseline_metrics["exact_match_rate"],
                latency_change_ratio=(quantized_metrics["latency_ms"] - baseline_metrics["latency_ms"]) / max(baseline_metrics["latency_ms"], 1),
                sensitivity_score=sensitivity,
            )

            layer_results.append(result)

            print(f"  Keyword Hit Rate: {quantized_metrics['keyword_hit_rate']:.2%} (Δ{result.keyword_hit_delta:+.2%})")
            print(f"  Sensitivity: {sensitivity:.4f}")

            del model_copy
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 生成报告
    report = SensitivityTestReport(
        model_name=args.model_dir,
        test_date=time.strftime("%Y-%m-%d %H:%M:%S"),
        num_samples=args.num_samples,
        method=args.method,
        baseline_keyword_hit=baseline_metrics["keyword_hit_rate"],
        baseline_exact_match=baseline_metrics["exact_match_rate"],
        baseline_latency_ms=baseline_metrics["latency_ms"],
        layer_results=[asdict(r) for r in layer_results],
        sensitive_layers=[r.layer_name for r in sorted(layer_results, key=lambda x: x.sensitivity_score, reverse=True)[:10]],
        recommendations=generate_recommendations(layer_results),
    )

    return report


def run_component_wise_test(args) -> SensitivityTestReport:
    """按组件量化敏感度测试"""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    import torch.quantization as quantization

    print("=" * 60)
    print("组件量化敏感度测试")
    print("=" * 60)

    # 加载验证数据
    val_data = load_vrsbench_data(args.data_dir)
    print(f"验证集样本数: {len(val_data)}")

    # 加载基座模型
    print("\n加载基座模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

    # 评估基线
    print("\n评估基线模型...")
    baseline_metrics = evaluate_model(model, processor, val_data, args.image_dir, args.num_samples)
    print(f"  Keyword Hit Rate: {baseline_metrics['keyword_hit_rate']:.2%}")

    # 定义组件测试 (仅测试不会导致 segfault 的组件)
    # 注意: 视觉编码器量化会导致 segfault，跳过
    components = {
        "language_model_only": "仅量化语言模型 Linear 层",
        "full_model": "量化所有 Linear 层",
    }

    layer_results = []

    for comp_name, description in components.items():
        print(f"\n测试组件: {comp_name} ({description})")

        try:
            # 创建模型副本
            model_copy = copy.deepcopy(model)

            # 量化
            if comp_name == "language_model_only":
                # 仅量化语言模型的 Linear 层
                for name, module in model_copy.named_modules():
                    if "language_model" in name and isinstance(module, nn.Linear):
                        quantized = quantization.quantize_dynamic(
                            module, {nn.Linear}, torch.qint8
                        )
                        # 替换原层
                        parts = name.split(".")
                        parent = model_copy
                        for part in parts[:-1]:
                            if part.isdigit():
                                parent = parent[int(part)]
                            else:
                                parent = getattr(parent, part)
                        setattr(parent, parts[-1], quantized)
            else:
                # 量化所有 Linear 层 (跳过视觉编码器)
                for name, module in model_copy.named_modules():
                    if isinstance(module, nn.Linear) and "visual" not in name:
                        quantized = quantization.quantize_dynamic(
                            module, {nn.Linear}, torch.qint8
                        )
                        # 替换原层
                        parts = name.split(".")
                        parent = model_copy
                        for part in parts[:-1]:
                            if part.isdigit():
                                parent = parent[int(part)]
                            else:
                                parent = getattr(parent, part)
                        setattr(parent, parts[-1], quantized)

            model_copy.eval()

            # 评估
            quantized_metrics = evaluate_model(model_copy, processor, val_data, args.image_dir, args.num_samples)

            # 计算敏感度
            sensitivity = calculate_sensitivity_score(baseline_metrics, quantized_metrics)

            # 统计参数量
            if comp_name == "language_model_only":
                num_params = sum(p.numel() for n, p in model_copy.named_parameters() if "language_model" in n)
            else:
                num_params = sum(p.numel() for n, p in model_copy.named_parameters() if "visual" not in n)

            result = LayerSensitivityResult(
                layer_name=comp_name,
                layer_type="Component",
                num_params=num_params,
                baseline_keyword_hit=baseline_metrics["keyword_hit_rate"],
                baseline_exact_match=baseline_metrics["exact_match_rate"],
                baseline_latency_ms=baseline_metrics["latency_ms"],
                quantized_keyword_hit=quantized_metrics["keyword_hit_rate"],
                quantized_exact_match=quantized_metrics["exact_match_rate"],
                quantized_latency_ms=quantized_metrics["latency_ms"],
                keyword_hit_delta=quantized_metrics["keyword_hit_rate"] - baseline_metrics["keyword_hit_rate"],
                exact_match_delta=quantized_metrics["exact_match_rate"] - baseline_metrics["exact_match_rate"],
                latency_change_ratio=(quantized_metrics["latency_ms"] - baseline_metrics["latency_ms"]) / max(baseline_metrics["latency_ms"], 1),
                sensitivity_score=sensitivity,
            )

            layer_results.append(result)

            print(f"  Keyword Hit Rate: {quantized_metrics['keyword_hit_rate']:.2%} (Δ{result.keyword_hit_delta:+.2%})")
            print(f"  Sensitivity: {sensitivity:.4f}")

            del model_copy
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 生成报告
    report = SensitivityTestReport(
        model_name=args.model_dir,
        test_date=time.strftime("%Y-%m-%d %H:%M:%S"),
        num_samples=args.num_samples,
        method=args.method,
        baseline_keyword_hit=baseline_metrics["keyword_hit_rate"],
        baseline_exact_match=baseline_metrics["exact_match_rate"],
        baseline_latency_ms=baseline_metrics["latency_ms"],
        layer_results=[asdict(r) for r in layer_results],
        sensitive_layers=[r.layer_name for r in sorted(layer_results, key=lambda x: x.sensitivity_score, reverse=True)[:5]],
        recommendations=generate_recommendations(layer_results),
    )

    return report


def save_report(report: SensitivityTestReport, output_dir: str):
    """保存报告"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存 JSON 报告
    report_file = os.path.join(output_dir, "sensitivity_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)

    # 保存 Markdown 报告
    md_file = os.path.join(output_dir, "sensitivity_report.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# 量化敏感度测试报告\n\n")
        f.write(f"**测试日期**: {report.test_date}\n\n")
        f.write(f"**测试模型**: {report.model_name}\n\n")
        f.write(f"**测试样本数**: {report.num_samples}\n\n")
        f.write(f"**测试方法**: {report.method}\n\n")

        f.write("## 基线结果\n\n")
        f.write(f"| 指标 | 数值 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| Keyword Hit Rate | {report.baseline_keyword_hit:.2%} |\n")
        f.write(f"| Exact Match Rate | {report.baseline_exact_match:.2%} |\n")
        f.write(f"| Latency | {report.baseline_latency_ms:.0f} ms |\n\n")

        f.write("## 各层敏感度结果\n\n")
        f.write("| 层名称 | 类型 | 参数量 | Keyword Hit Δ | 敏感度 |\n")
        f.write("|--------|------|--------|---------------|--------|\n")

        sorted_results = sorted(report.layer_results, key=lambda x: x["sensitivity_score"], reverse=True)
        for r in sorted_results:
            f.write(f"| {r['layer_name'][:40]}... | {r['layer_type']} | {r['num_params']:,} | {r['keyword_hit_delta']:+.2%} | {r['sensitivity_score']:.4f} |\n")

        f.write("\n## 高敏感层 (Top 10)\n\n")
        for i, layer_name in enumerate(report.sensitive_layers[:10], 1):
            f.write(f"{i}. `{layer_name}`\n")

        f.write("\n## 建议\n\n")
        for rec in report.recommendations:
            f.write(f"- {rec}\n")

    print(f"\n报告已保存:")
    print(f"  JSON: {report_file}")
    print(f"  Markdown: {md_file}")


def main():
    parser = argparse.ArgumentParser(description="层量化敏感度测试")
    parser.add_argument("--model-dir", required=True, help="模型目录")
    parser.add_argument("--data-dir", required=True, help="VRSBench 数据目录")
    parser.add_argument("--image-dir", required=True, help="图片目录")
    parser.add_argument("--output-dir", default="reports/quantization_sensitivity", help="输出目录")
    parser.add_argument("--num-samples", type=int, default=20, help="评估样本数")
    parser.add_argument("--method", default="layer_wise", choices=["layer_wise", "component_wise"], help="测试方法")
    args = parser.parse_args()

    print(f"模型目录: {args.model_dir}")
    print(f"数据目录: {args.data_dir}")
    print(f"图片目录: {args.image_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"测试方法: {args.method}")

    # 运行测试
    if args.method == "layer_wise":
        report = run_layer_wise_test(args)
    else:
        report = run_component_wise_test(args)

    # 保存报告
    save_report(report, args.output_dir)

    print("\n测试完成!")


if __name__ == "__main__":
    main()
