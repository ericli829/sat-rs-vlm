"""
对已合并的 LoRA 模型进行 INT8 量化

直接从保存的合并模型加载，避免重复的合并操作。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

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
    parser = argparse.ArgumentParser(description="量化已合并的模型")
    parser.add_argument("--merged-model", required=True, help="已合并模型路径")
    parser.add_argument("--output-dir", required=True, help="量化模型输出路径")
    parser.add_argument("--base-model", default="D:/Models/Qwen3-VL-2B-Instruct", help="基座模型路径（用于加载processor）")
    args = parser.parse_args()

    print(f"PyTorch 版本: {torch.__version__}")
    print(f"已合并模型: {args.merged_model}")
    print(f"输出目录: {args.output_dir}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # ==================== 步骤1: 加载已合并模型 ====================
    print("\n" + "="*60)
    print("步骤1: 加载已合并模型")
    print("="*60)

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    merged_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.merged_model,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    merged_model.eval()

    merged_params = count_parameters(merged_model)
    print(f"合并后模型参数量: {merged_params:,}")

    # ==================== 步骤2: INT8 量化 ====================
    print("\n" + "="*60)
    print("步骤2: 执行 INT8 动态量化")
    print("="*60)

    # 统计原始模型中的 Linear 层
    linear_count = sum(1 for _, m in merged_model.named_modules() if isinstance(m, torch.nn.Linear))
    print(f"Linear 层: {linear_count}")

    # 执行量化
    print("执行量化...")
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            merged_model.cpu(),
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        print("量化成功!")
    except Exception as e:
        print(f"量化失败: {e}")
        return

    quantized_params = count_parameters(quantized_model)
    print(f"量化后模型参数量: {quantized_params:,}")

    # ==================== 步骤3: 保存量化模型 ====================
    print("\n" + "="*60)
    print("步骤3: 保存量化模型")
    print("="*60)

    # 保存模型状态字典
    model_path = os.path.join(args.output_dir, "model.pt")
    print(f"保存模型到: {model_path}")
    torch.save(quantized_model.state_dict(), model_path)

    # 保存 processor
    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    processor.save_pretrained(args.output_dir)

    # 保存配置
    config = {
        "base_model": args.base_model,
        "merged_model": args.merged_model,
        "quantization_method": "INT8_Dynamic_CPU",
        "quantization_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "merged_params": merged_params,
        "quantized_params": quantized_params,
    }

    config_path = os.path.join(args.output_dir, "quantization_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 计算大小
    model_file_size = os.path.getsize(model_path) / (1024 ** 3)
    total_size = get_model_size_gb(args.output_dir)

    # ==================== 步骤4: 验证量化模型 ====================
    print("\n" + "="*60)
    print("步骤4: 验证量化模型")
    print("="*60)

    print("测试加载量化模型...")
    test_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    test_model.load_state_dict(torch.load(model_path, weights_only=True))
    test_model.eval()
    print("量化模型加载验证成功!")

    # ==================== 生成报告 ====================
    print("\n" + "="*60)
    print("量化结果汇总")
    print("="*60)

    report = {
        "model_name": Path(args.base_model).name + "-LoRA-INT8",
        "base_model": args.base_model,
        "merged_model": args.merged_model,
        "quantization_method": "INT8_Dynamic_CPU",
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "merged_model": merged_params,
            "quantized_model": quantized_params,
        },
        "model_size": {
            "merged_model_gb": get_model_size_gb(args.merged_model),
            "model_pt_gb": model_file_size,
            "total_output_gb": total_size,
        },
        "output_dir": args.output_dir,
    }

    report_path = os.path.join(args.output_dir, "quantization_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n已合并模型: {args.merged_model}")
    print(f"输出目录: {args.output_dir}")

    print(f"\n参数量统计:")
    print(f"  合并后模型: {merged_params:,}")
    print(f"  量化后模型: {quantized_params:,}")

    print(f"\n模型大小:")
    print(f"  合并后模型: {report['model_size']['merged_model_gb']:.2f} GB")
    print(f"  量化模型文件: {model_file_size:.2f} GB")
    print(f"  输出目录总大小: {total_size:.2f} GB")

    print(f"\n报告已保存至: {report_path}")

    return report


if __name__ == "__main__":
    main()
