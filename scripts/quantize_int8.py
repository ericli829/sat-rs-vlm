"""
INT8 动态量化脚本

将 Qwen3-VL-2B-Instruct 模型进行 INT8 动态量化，
测量量化前后的模型大小、推理速度和显存占用。

用法：
    python scripts/quantize_int8.py \
        --model-dir D:\Models\Qwen3-VL-2B-Instruct \
        --output-dir checkpoints/quantized/int8 \
        --val-jsonl data/processed/qwen3vl_val.jsonl \
        --image-root F:\VIT-data\VRSBench \
        --num-samples 50 \
        --warmup-samples 5
"""

import argparse
import json
import os
import time
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import numpy as np
from PIL import Image


@dataclass
class QuantizationResult:
    """量化结果"""
    model_name: str
    # 原始模型信息
    original_size_gb: float = 0.0
    original_dtype: str = ""
    # 量化后信息
    quantized_size_gb: float = 0.0
    quantization_method: str = "INT8_Dynamic"
    # 推理性能对比
    original_inference_ms: float = 0.0
    quantized_inference_ms: float = 0.0
    speedup_ratio: float = 0.0
    # 显存对比
    original_vram_gb: float = 0.0
    quantized_vram_gb: float = 0.0
    vram_saving_ratio: float = 0.0
    # 参数信息
    original_params: int = 0
    quantized_params: int = 0
    # 评估结果
    original_accuracy: float = 0.0
    quantized_accuracy: float = 0.0
    accuracy_retention: float = 0.0
    # 输出路径
    output_dir: str = ""
    report_file: str = ""
    evaluation_time: str = ""


def get_model_size_gb(model_dir: str) -> float:
    """计算模型目录大小"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except OSError:
                continue
    return total_size / (1024 ** 3)


def measure_inference_speed(
    model,
    processor,
    test_images: list[str],
    prompt: str = "请描述这张遥感图像。",
    num_runs: int = 3,
    device: str = "cuda",
) -> dict:
    """测量推理速度"""
    times = []

    for _ in range(num_runs):
        for img_path in test_images:
            try:
                image = Image.open(img_path).convert("RGB")

                # 构造输入
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[text], images=[image], return_tensors="pt"
                )

                # 移动到模型设备
                inputs = {
                    k: v.to(device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

                # 同步
                if device == "cuda":
                    torch.cuda.synchronize()

                # 推理计时
                start_time = time.perf_counter()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=128, use_cache=True
                    )
                end_time = time.perf_counter()

                inference_time = (end_time - start_time) * 1000
                times.append(inference_time)

            except Exception as e:
                print(f"  警告: 推理失败 {img_path}: {e}")
                continue

    if not times:
        return {"avg_ms": 0, "std_ms": 0, "min_ms": 0, "max_ms": 0}

    return {
        "avg_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def measure_vram_usage(model, processor, test_image: str, device: str = "cuda") -> dict:
    """测量显存使用"""
    if device != "cuda":
        return {"peak_vram_gb": 0, "allocated_vram_gb": 0}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    image = Image.open(test_image).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "请描述这张遥感图像。"},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)

    torch.cuda.synchronize()
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    allocated_vram = torch.cuda.memory_allocated() / (1024 ** 3)

    return {
        "peak_vram_gb": peak_vram,
        "allocated_vram_gb": allocated_vram,
    }


def compute_accuracy(
    model, processor, val_data: list[dict], image_root: str,
    device: str = "cuda", max_samples: int = 50
) -> float:
    """计算模型准确率"""
    correct = 0
    total = 0

    for item in val_data[:max_samples]:
        try:
            image_path = os.path.join(image_root, item.get("image", ""))
            if not os.path.exists(image_path):
                continue

            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": item.get("instruction", "")},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=[image], return_tensors="pt")
            inputs = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=128)
                pred_text = processor.decode(outputs[0], skip_special_tokens=True)

            ref_answer = item.get("answer", "").strip().lower()
            pred_answer = pred_text.strip().lower()

            if ref_answer in pred_answer or pred_answer in ref_answer:
                correct += 1
            total += 1

        except Exception as e:
            continue

    return correct / total if total > 0 else 0.0


def load_int8_model(model_dir: str):
    """加载 INT8 量化模型"""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    print(f"加载 INT8 量化模型: {model_dir}")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    return model


def load_baseline_model(model_dir: str):
    """加载基线模型（BF16）"""
    from transformers import AutoModelForCausalLM

    print(f"加载基线模型: {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


def save_int8_model(model, processor, output_dir: str):
    """保存 INT8 量化模型"""
    print(f"保存 INT8 模型到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 保存模型（BitsAndBytes 量化后的模型可以直接保存）
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)

    print(f"模型保存完成")


def main():
    parser = argparse.ArgumentParser(description="INT8 动态量化")
    parser.add_argument("--model-dir", required=True, help="原始模型路径")
    parser.add_argument("--output-dir", required=True, help="量化模型输出路径")
    parser.add_argument("--val-jsonl", required=True, help="验证集JSONL")
    parser.add_argument("--image-root", required=True, help="图片根目录")
    parser.add_argument("--num-samples", type=int, default=50, help="评估样本数")
    parser.add_argument("--warmup-samples", type=int, default=5, help="预热样本数")
    parser.add_argument("--skip-baseline", action="store_true", help="跳过基线评估")
    parser.add_argument("--skip-save", action="store_true", help="跳过保存模型")
    args = parser.parse_args()

    # 检查 CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # 初始化结果
    result = QuantizationResult(
        model_name=Path(args.model_dir).name,
        quantization_method="INT8_Dynamic",
        output_dir=args.output_dir,
        evaluation_time=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # 计算原始模型大小
    result.original_size_gb = get_model_size_gb(args.model_dir)
    print(f"\n原始模型大小: {result.original_size_gb:.2f} GB")

    # 加载验证数据
    val_data = []
    with open(args.val_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            val_data.append(json.loads(line))
    print(f"验证集样本数: {len(val_data)}")

    # 收集测试图像
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    test_images = []
    for f in os.listdir(args.image_root):
        if Path(f).suffix.lower() in image_extensions:
            test_images.append(os.path.join(args.image_root, f))
            if len(test_images) >= args.num_samples + args.warmup_samples:
                break
    print(f"测试图像数: {len(test_images)}")

    # ==================== 基线评估 ====================
    if not args.skip_baseline:
        print("\n" + "="*60)
        print("基线模型评估 (BF16)")
        print("="*60)

        baseline_model = load_baseline_model(args.model_dir)
        from transformers import AutoProcessor
        baseline_processor = AutoProcessor.from_pretrained(
            args.model_dir, trust_remote_code=True
        )
        baseline_model.eval()

        # 基线参数量
        result.original_params = sum(
            p.numel() for p in baseline_model.parameters()
        )
        print(f"基线参数量: {result.original_params:,}")

        # 基线显存
        if device == "cuda":
            print("测量基线显存...")
            vram_stats = measure_vram_usage(
                baseline_model, baseline_processor,
                test_images[0], device
            )
            result.original_vram_gb = vram_stats["peak_vram_gb"]
            print(f"基线峰值显存: {result.original_vram_gb:.2f} GB")

        # 基线推理速度
        print("测量基线推理速度...")
        warmup_images = test_images[:args.warmup_samples]
        test_image_list = test_images[args.warmup_samples:args.num_samples + args.warmup_samples]

        # 预热
        measure_inference_speed(
            baseline_model, baseline_processor,
            warmup_images, num_runs=1, device=device
        )

        # 正式测量
        baseline_speed = measure_inference_speed(
            baseline_model, baseline_processor,
            test_image_list, num_runs=3, device=device
        )
        result.original_inference_ms = baseline_speed["avg_ms"]
        print(f"基线推理速度: {result.original_inference_ms:.1f} ms "
              f"(±{baseline_speed['std_ms']:.1f} ms)")

        # 基线准确率
        print("评估基线准确率...")
        result.original_accuracy = compute_accuracy(
            baseline_model, baseline_processor,
            val_data, args.image_root, device, args.num_samples
        )
        print(f"基线准确率: {result.original_accuracy:.4f}")

        # 清理基线模型
        del baseline_model
        del baseline_processor
        if device == "cuda":
            torch.cuda.empty_cache()

    # ==================== INT8 量化 ====================
    print("\n" + "="*60)
    print("INT8 量化模型评估")
    print("="*60)

    quantized_model = load_int8_model(args.model_dir)
    from transformers import AutoProcessor
    quantized_processor = AutoProcessor.from_pretrained(
        args.model_dir, trust_remote_code=True
    )
    quantized_model.eval()

    # 量化模型参数量
    result.quantized_params = sum(
        p.numel() for p in quantized_model.parameters()
    )
    print(f"量化模型参数量: {result.quantized_params:,}")

    # 量化模型显存
    if device == "cuda":
        print("测量量化模型显存...")
        vram_stats = measure_vram_usage(
            quantized_model, quantized_processor,
            test_images[0], device
        )
        result.quantized_vram_gb = vram_stats["peak_vram_gb"]
        result.vram_saving_ratio = (
            1 - result.quantized_vram_gb / result.original_vram_gb
            if result.original_vram_gb > 0 else 0
        )
        print(f"量化模型峰值显存: {result.quantized_vram_gb:.2f} GB")
        print(f"显存节省: {result.vram_saving_ratio:.2%}")

    # 量化模型推理速度
    print("测量量化模型推理速度...")
    # 预热
    measure_inference_speed(
        quantized_model, quantized_processor,
        warmup_images, num_runs=1, device=device
    )

    # 正式测量
    quantized_speed = measure_inference_speed(
        quantized_model, quantized_processor,
        test_image_list, num_runs=3, device=device
    )
    result.quantized_inference_ms = quantized_speed["avg_ms"]
    result.speedup_ratio = (
        result.original_inference_ms / result.quantized_inference_ms
        if result.quantized_inference_ms > 0 else 0
    )
    print(f"量化模型推理速度: {result.quantized_inference_ms:.1f} ms "
          f"(±{quantized_speed['std_ms']:.1f} ms)")
    print(f"加速比: {result.speedup_ratio:.2f}×")

    # 量化模型准确率
    print("评估量化模型准确率...")
    result.quantized_accuracy = compute_accuracy(
        quantized_model, quantized_processor,
        val_data, args.image_root, device, args.num_samples
    )
    result.accuracy_retention = (
        result.quantized_accuracy / result.original_accuracy
        if result.original_accuracy > 0 else 0
    )
    print(f"量化模型准确率: {result.quantized_accuracy:.4f}")
    print(f"精度保持率: {result.accuracy_retention:.2%}")

    # ==================== 保存量化模型 ====================
    if not args.skip_save:
        print("\n" + "="*60)
        print("保存 INT8 量化模型")
        print("="*60)
        save_int8_model(quantized_model, quantized_processor, args.output_dir)

        # 计算保存后的模型大小
        result.quantized_size_gb = get_model_size_gb(args.output_dir)
        print(f"保存后模型大小: {result.quantized_size_gb:.2f} GB")

    # ==================== 生成报告 ====================
    print("\n" + "="*60)
    print("量化结果汇总")
    print("="*60)

    # 保存报告
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / "int8_quantization_report.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    result.report_file = str(report_file)

    # 打印摘要
    print(f"\n模型: {result.model_name}")
    print(f"量化方法: {result.quantization_method}")
    print(f"\n模型大小:")
    print(f"  原始: {result.original_size_gb:.2f} GB")
    print(f"  量化后: {result.quantized_size_gb:.2f} GB")
    if result.original_size_gb > 0:
        compression_ratio = result.original_size_gb / result.quantized_size_gb if result.quantized_size_gb > 0 else 0
        print(f"  压缩比: {compression_ratio:.2f}×")

    print(f"\n参数量:")
    print(f"  原始: {result.original_params:,}")
    print(f"  量化后: {result.quantized_params:,}")

    print(f"\n推理速度:")
    print(f"  原始: {result.original_inference_ms:.1f} ms")
    print(f"  量化后: {result.quantized_inference_ms:.1f} ms")
    print(f"  加速比: {result.speedup_ratio:.2f}×")

    print(f"\n显存占用:")
    print(f"  原始: {result.original_vram_gb:.2f} GB")
    print(f"  量化后: {result.quantized_vram_gb:.2f} GB")
    print(f"  节省: {result.vram_saving_ratio:.2%}")

    print(f"\n准确率:")
    print(f"  原始: {result.original_accuracy:.4f}")
    print(f"  量化后: {result.quantized_accuracy:.4f}")
    print(f"  保持率: {result.accuracy_retention:.2%}")

    print(f"\n报告已保存至: {report_file}")

    return result


if __name__ == "__main__":
    main()
