"""
LoRA 模型 INT8 CPU 量化脚本

将训练好的 LoRA adapter 合并到基座模型，然后进行 INT8 动态量化。
量化后的模型保存到 D:\models 目录下，不影响原有模型。

用法：
    python scripts/quantize_lora_int8_cpu.py \
        --base-model D:\Models\Qwen3-VL-2B-Instruct \
        --adapter-path checkpoints/lora/autodl_4090_full \
        --output-dir D:\models\Qwen3-VL-2B-Instruct-LoRA-INT8 \
        --val-jsonl data/processed/qwen3vl_val.jsonl \
        --image-root d:\project\database\VRSBench \
        --num-samples 20 \
        --warmup-samples 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import numpy as np
from PIL import Image

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass
class QuantizationResult:
    """量化结果"""
    model_name: str
    base_model: str = ""
    adapter_path: str = ""
    # 原始模型信息
    original_size_gb: float = 0.0
    original_dtype: str = ""
    original_params: int = 0
    # LoRA 模型信息
    lora_params: int = 0
    trainable_params: int = 0
    # 量化后信息
    quantized_size_gb: float = 0.0
    quantization_method: str = "LoRA_Merged_INT8_Dynamic_CPU"
    quantized_params: int = 0
    # 推理性能对比
    lora_inference_ms: float = 0.0
    quantized_inference_ms: float = 0.0
    speedup_ratio: float = 0.0
    # 模型大小对比
    compression_ratio: float = 0.0
    # 评估结果
    lora_accuracy: float = 0.0
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


def extract_from_messages(item: dict, image_root: str) -> tuple[str, str, str] | None:
    """
    从 Qwen3-VL messages 格式提取图像路径、问题和答案
    """
    messages = item.get("messages", [])
    if len(messages) < 2:
        return None

    user_msg = messages[0]
    if user_msg.get("role") != "user":
        return None

    content = user_msg.get("content", [])
    image_path = None
    question = None

    for c in content:
        if c.get("type") == "image":
            img_rel = c.get("image", "")
            image_path = os.path.join(image_root, img_rel)
        elif c.get("type") == "text":
            question = c.get("text", "")

    assistant_msg = messages[1]
    answer = assistant_msg.get("content", "")

    if image_path and question and answer:
        return (image_path, question, answer)
    return None


def compute_accuracy(
    model, processor, val_data: list[dict], image_root: str,
    max_samples: int = 20
) -> float:
    """计算模型准确率"""
    correct = 0
    total = 0

    for item in val_data[:max_samples]:
        try:
            extracted = extract_from_messages(item, image_root)
            if not extracted:
                continue

            image_path, question, ref_answer = extracted
            if not os.path.exists(image_path):
                continue

            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=[image], return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=64)
                pred_text = processor.decode(outputs[0], skip_special_tokens=True)

            pred_answer = pred_text.strip().lower()
            ref_answer = ref_answer.strip().lower()

            # 对于长答案，检查关键内容是否匹配
            if len(ref_answer) > 20:
                ref_words = set(ref_answer.split())
                pred_words = set(pred_answer.split())
                overlap = len(ref_words & pred_words) / max(len(ref_words), 1)
                if overlap > 0.5:
                    correct += 1
            else:
                # 短答案：精确匹配或包含
                if ref_answer in pred_answer or pred_answer in ref_answer:
                    correct += 1
            total += 1

        except Exception as e:
            print(f"  警告: 评估失败: {e}")
            continue

    return correct / total if total > 0 else 0.0


def measure_inference_speed(
    model, processor, test_images: list[str],
    val_data: list[dict], image_root: str,
    num_runs: int = 2,
) -> dict:
    """测量推理速度"""
    times = []

    questions = {}
    for item in val_data:
        extracted = extract_from_messages(item, image_root)
        if extracted:
            img_path, question, _ = extracted
            if img_path not in questions:
                questions[img_path] = question

    for _ in range(num_runs):
        for img_path in test_images:
            try:
                if not os.path.exists(img_path):
                    continue

                image = Image.open(img_path).convert("RGB")
                prompt = questions.get(img_path, "请描述这张遥感图像。")

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

                start_time = time.perf_counter()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=64, use_cache=True
                    )
                end_time = time.perf_counter()

                inference_time = (end_time - start_time) * 1000
                times.append(inference_time)

            except Exception as e:
                print(f"  警告: 推理失败 {Path(img_path).name}: {e}")
                continue

    if not times:
        return {"avg_ms": 0, "std_ms": 0, "min_ms": 0, "max_ms": 0}

    return {
        "avg_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def load_lora_model(base_model_path: str, adapter_path: str):
    """加载 LoRA 模型"""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    print(f"加载基座模型: {base_model_path}")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"加载 LoRA adapter: {adapter_path}")
    lora_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        local_files_only=True,
    )

    processor = AutoProcessor.from_pretrained(
        base_model_path, trust_remote_code=True
    )

    return lora_model, processor


def merge_and_unload(lora_model):
    """合并 LoRA 权重到基座模型"""
    print("合并 LoRA 权重到基座模型...")
    merged_model = lora_model.merge_and_unload()
    return merged_model


def quantize_model_dynamic_int8(model):
    """使用 PyTorch 原生动态 INT8 量化"""
    print("执行动态 INT8 量化...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


def save_quantized_model(model, processor, output_dir: str):
    """保存量化模型"""
    print(f"保存量化模型到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        model.save_pretrained(output_dir, safe_serialization=True)
    except Exception as e:
        print(f"safetensors 保存失败，尝试 torch.save: {e}")
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    processor.save_pretrained(output_dir)
    print(f"模型保存完成")


def main():
    parser = argparse.ArgumentParser(description="LoRA 模型 INT8 CPU 量化")
    parser.add_argument("--base-model", required=True, help="基座模型路径")
    parser.add_argument("--adapter-path", required=True, help="LoRA adapter 路径")
    parser.add_argument("--output-dir", required=True, help="量化模型输出路径")
    parser.add_argument("--val-jsonl", required=True, help="验证集JSONL")
    parser.add_argument("--image-root", required=True, help="图片根目录")
    parser.add_argument("--num-samples", type=int, default=20, help="评估样本数")
    parser.add_argument("--warmup-samples", type=int, default=3, help="预热样本数")
    parser.add_argument("--skip-baseline", action="store_true", help="跳过基线评估")
    parser.add_argument("--skip-save", action="store_true", help="跳过保存模型")
    args = parser.parse_args()

    print(f"使用设备: CPU")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"基座模型: {args.base_model}")
    print(f"LoRA adapter: {args.adapter_path}")
    print(f"输出目录: {args.output_dir}")

    # 初始化结果
    result = QuantizationResult(
        model_name=Path(args.base_model).name + "-LoRA-INT8",
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        quantization_method="LoRA_Merged_INT8_Dynamic_CPU",
        output_dir=args.output_dir,
        evaluation_time=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # 计算原始模型大小
    result.original_size_gb = get_model_size_gb(args.base_model)
    print(f"\n基座模型大小: {result.original_size_gb:.2f} GB")

    # 加载验证数据
    val_data = []
    with open(args.val_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            val_data.append(json.loads(line))
    print(f"验证集样本数: {len(val_data)}")

    # 收集测试图像
    test_images = []
    for item in val_data:
        extracted = extract_from_messages(item, args.image_root)
        if extracted:
            img_path = extracted[0]
            if os.path.exists(img_path) and img_path not in test_images:
                test_images.append(img_path)
                if len(test_images) >= args.num_samples + args.warmup_samples:
                    break
    print(f"测试图像数: {len(test_images)}")

    # ==================== 加载 LoRA 模型 ====================
    print("\n" + "="*60)
    print("加载 LoRA 模型")
    print("="*60)

    lora_model, processor = load_lora_model(args.base_model, args.adapter_path)
    lora_model.eval()

    # LoRA 模型参数量
    result.original_params = sum(p.numel() for p in lora_model.parameters())
    result.trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"LoRA 模型总参数量: {result.original_params:,}")
    print(f"可训练参数量: {result.trainable_params:,}")

    # LoRA 模型推理速度
    print("\n测量 LoRA 模型推理速度...")
    warmup_images = test_images[:args.warmup_samples]
    test_image_list = test_images[args.warmup_samples:args.num_samples + args.warmup_samples]

    # 预热
    measure_inference_speed(
        lora_model, processor,
        warmup_images, val_data, args.image_root, num_runs=1
    )

    # 正式测量
    lora_speed = measure_inference_speed(
        lora_model, processor,
        test_image_list, val_data, args.image_root, num_runs=2
    )
    result.lora_inference_ms = lora_speed["avg_ms"]
    print(f"LoRA 模型推理速度: {result.lora_inference_ms:.1f} ms "
          f"(±{lora_speed['std_ms']:.1f} ms)")

    # LoRA 模型准确率
    print("\n评估 LoRA 模型准确率...")
    result.lora_accuracy = compute_accuracy(
        lora_model, processor,
        val_data, args.image_root, args.num_samples
    )
    print(f"LoRA 模型准确率: {result.lora_accuracy:.4f}")

    # ==================== 合并 LoRA 权重 ====================
    print("\n" + "="*60)
    print("合并 LoRA 权重到基座模型")
    print("="*60)

    merged_model = merge_and_unload(lora_model)
    merged_model.eval()

    # 合并后参数量
    merged_params = sum(p.numel() for p in merged_model.parameters())
    print(f"合并后模型参数量: {merged_params:,}")

    # ==================== INT8 量化 ====================
    print("\n" + "="*60)
    print("执行 INT8 量化")
    print("="*60)

    quantized_model = quantize_model_dynamic_int8(merged_model)

    # 量化模型参数量
    result.quantized_params = sum(
        p.numel() for p in quantized_model.parameters()
    )
    print(f"量化模型参数量: {result.quantized_params:,}")

    # 量化模型推理速度
    print("\n测量量化模型推理速度...")
    measure_inference_speed(
        quantized_model, processor,
        warmup_images, val_data, args.image_root, num_runs=1
    )

    quantized_speed = measure_inference_speed(
        quantized_model, processor,
        test_image_list, val_data, args.image_root, num_runs=2
    )
    result.quantized_inference_ms = quantized_speed["avg_ms"]
    result.speedup_ratio = (
        result.lora_inference_ms / result.quantized_inference_ms
        if result.quantized_inference_ms > 0 else 0
    )
    print(f"量化模型推理速度: {result.quantized_inference_ms:.1f} ms "
          f"(±{quantized_speed['std_ms']:.1f} ms)")
    print(f"加速比: {result.speedup_ratio:.2f}×")

    # 量化模型准确率
    print("\n评估量化模型准确率...")
    result.quantized_accuracy = compute_accuracy(
        quantized_model, processor,
        val_data, args.image_root, args.num_samples
    )
    result.accuracy_retention = (
        result.quantized_accuracy / result.lora_accuracy
        if result.lora_accuracy > 0 else 0
    )
    print(f"量化模型准确率: {result.quantized_accuracy:.4f}")
    print(f"精度保持率: {result.accuracy_retention:.2%}")

    # ==================== 保存量化模型 ====================
    if not args.skip_save:
        print("\n" + "="*60)
        print("保存 INT8 量化模型")
        print("="*60)
        save_quantized_model(quantized_model, processor, args.output_dir)

        result.quantized_size_gb = get_model_size_gb(args.output_dir)
        result.compression_ratio = (
            result.original_size_gb / result.quantized_size_gb
            if result.quantized_size_gb > 0 else 0
        )
        print(f"保存后模型大小: {result.quantized_size_gb:.2f} GB")
        print(f"压缩比: {result.compression_ratio:.2f}×")

    # ==================== 生成报告 ====================
    print("\n" + "="*60)
    print("量化结果汇总")
    print("="*60)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / "int8_quantization_report.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    result.report_file = str(report_file)

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"模型: {result.model_name}")
    print(f"量化方法: {result.quantization_method}")
    print(f"{'='*60}")

    print(f"\n基座模型: {result.base_model}")
    print(f"LoRA adapter: {result.adapter_path}")

    print(f"\n模型大小:")
    print(f"  基座模型: {result.original_size_gb:.2f} GB")
    print(f"  量化后: {result.quantized_size_gb:.2f} GB")
    if result.compression_ratio > 0:
        print(f"  压缩比: {result.compression_ratio:.2f}×")

    print(f"\n参数量:")
    print(f"  LoRA 模型总参数: {result.original_params:,}")
    print(f"  可训练参数: {result.trainable_params:,}")
    print(f"  量化后参数: {result.quantized_params:,}")

    print(f"\n推理速度 (CPU):")
    print(f"  LoRA 模型: {result.lora_inference_ms:.1f} ms")
    print(f"  量化模型: {result.quantized_inference_ms:.1f} ms")
    if result.speedup_ratio > 0:
        print(f"  加速比: {result.speedup_ratio:.2f}×")

    print(f"\n准确率:")
    print(f"  LoRA 模型: {result.lora_accuracy:.4f}")
    print(f"  量化模型: {result.quantized_accuracy:.4f}")
    print(f"  保持率: {result.accuracy_retention:.2%}")

    print(f"\n报告已保存至: {report_file}")

    return result


if __name__ == "__main__":
    main()
