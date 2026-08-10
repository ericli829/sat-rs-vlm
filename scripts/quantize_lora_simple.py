"""
简化的 LoRA 模型 INT8 量化脚本

分步骤进行量化，避免内存问题：
1. 加载并合并 LoRA 模型
2. 保存合并后的模型
3. 进行 INT8 量化
4. 评估量化效果
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


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


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="简化 LoRA INT8 量化")
    parser.add_argument("--base-model", required=True, help="基座模型路径")
    parser.add_argument("--adapter-path", required=True, help="LoRA adapter 路径")
    parser.add_argument("--output-dir", required=True, help="量化模型输出路径")
    parser.add_argument("--merged-dir", default=None, help="合并后模型保存路径（可选）")
    args = parser.parse_args()

    print(f"PyTorch 版本: {torch.__version__}")
    print(f"基座模型: {args.base_model}")
    print(f"LoRA adapter: {args.adapter_path}")
    print(f"输出目录: {args.output_dir}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    if args.merged_dir:
        os.makedirs(args.merged_dir, exist_ok=True)

    # ==================== 步骤1: 加载基座模型 ====================
    print("\n" + "="*60)
    print("步骤1: 加载基座模型")
    print("="*60)

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True
    )

    base_params = count_parameters(base_model)
    print(f"基座模型参数量: {base_params:,}")

    # ==================== 步骤2: 加载并合并 LoRA ====================
    print("\n" + "="*60)
    print("步骤2: 加载并合并 LoRA adapter")
    print("="*60)

    from peft import PeftModel

    lora_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        local_files_only=True,
    )

    lora_params = count_parameters(lora_model)
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"LoRA 模型总参数量: {lora_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"可训练参数比例: {trainable_params/lora_params*100:.2f}%")

    # 合并 LoRA 权重
    print("\n合并 LoRA 权重到基座模型...")
    merged_model = lora_model.merge_and_unload()
    del lora_model  # 释放内存

    merged_params = count_parameters(merged_model)
    print(f"合并后模型参数量: {merged_params:,}")

    # 保存合并后的模型（可选）
    if args.merged_dir:
        print(f"\n保存合并后模型到: {args.merged_dir}")
        merged_model.save_pretrained(args.merged_dir, safe_serialization=True)
        processor.save_pretrained(args.merged_dir)
        merged_size = get_model_size_gb(args.merged_dir)
        print(f"合并后模型大小: {merged_size:.2f} GB")

    # ==================== 步骤3: INT8 量化 ====================
    print("\n" + "="*60)
    print("步骤3: 执行 INT8 动态量化")
    print("="*60)

    # 统计原始模型中的 Linear 层
    linear_layers = []
    for name, module in merged_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append(name)
    print(f"Linear 层数量: {len(linear_layers)}")

    # 执行动态量化
    print("执行量化...")
    quantized_model = torch.quantization.quantize_dynamic(
        merged_model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    del merged_model  # 释放内存

    quantized_params = count_parameters(quantized_model)
    print(f"量化后模型参数量: {quantized_params:,}")

    # ==================== 步骤4: 保存量化模型 ====================
    print("\n" + "="*60)
    print("步骤4: 保存量化模型")
    print("="*60)

    # 保存模型
    model_path = os.path.join(args.output_dir, "model.pt")
    print(f"保存模型到: {model_path}")
    torch.save(quantized_model.state_dict(), model_path)

    # 保存 processor
    print("保存 processor...")
    processor.save_pretrained(args.output_dir)

    # 保存配置信息
    config = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "quantization_method": "INT8_Dynamic_CPU",
        "quantization_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "base_params": base_params,
        "lora_trainable_params": trainable_params,
        "merged_params": merged_params,
        "quantized_params": quantized_params,
    }

    config_path = os.path.join(args.output_dir, "quantization_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"保存配置到: {config_path}")

    # ==================== 步骤5: 验证量化模型 ====================
    print("\n" + "="*60)
    print("步骤5: 验证量化模型")
    print("="*60)

    # 测试加载量化模型
    print("测试加载量化模型...")
    test_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    test_model.load_state_dict(torch.load(model_path, weights_only=True))
    test_model.eval()
    print("量化模型加载成功!")

    # 计算模型大小
    model_file_size = os.path.getsize(model_path) / (1024 ** 3)
    total_size = get_model_size_gb(args.output_dir)

    # ==================== 生成报告 ====================
    print("\n" + "="*60)
    print("量化结果汇总")
    print("="*60)

    report = {
        "model_name": Path(args.base_model).name + "-LoRA-INT8",
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "quantization_method": "INT8_Dynamic_CPU",
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "base_model": base_params,
            "lora_trainable": trainable_params,
            "merged_model": merged_params,
            "quantized_model": quantized_params,
        },
        "model_size": {
            "base_model_gb": get_model_size_gb(args.base_model),
            "model_pt_gb": model_file_size,
            "total_output_gb": total_size,
        },
        "output_dir": args.output_dir,
    }

    report_path = os.path.join(args.output_dir, "quantization_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n基座模型: {args.base_model}")
    print(f"LoRA adapter: {args.adapter_path}")
    print(f"输出目录: {args.output_dir}")

    print(f"\n参数量统计:")
    print(f"  基座模型: {base_params:,}")
    print(f"  LoRA 可训练参数: {trainable_params:,}")
    print(f"  合并后模型: {merged_params:,}")
    print(f"  量化后模型: {quantized_params:,}")

    print(f"\n模型大小:")
    print(f"  基座模型: {report['model_size']['base_model_gb']:.2f} GB")
    print(f"  量化模型文件: {model_file_size:.2f} GB")
    print(f"  输出目录总大小: {total_size:.2f} GB")

    print(f"\n报告已保存至: {report_path}")

    return report


if __name__ == "__main__":
    main()
