"""
生成量化评测配置脚本

根据环境自动生成适合当前环境的量化评测配置文件。

用法：
    python scripts/generate_quantization_eval_config.py \
        --config configs/quantization/sensitivity_test.yaml \
        --environment autodl \
        --overwrite
"""

import argparse
import os
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_autodl_config(config: dict) -> dict:
    """生成 AutoDL 云端配置"""
    autodl_config = config.copy()

    # 更新模型路径
    if "model" in autodl_config and "autodl_dir" in autodl_config["model"]:
        autodl_config["model"]["local_dir"] = autodl_config["model"]["autodl_dir"]

    # 更新数据路径
    if "data" in autodl_config:
        if "autodl_metadata_dir" in autodl_config["data"]:
            autodl_config["data"]["local_metadata_dir"] = autodl_config["data"]["autodl_metadata_dir"]
        if "autodl_image_dir" in autodl_config["data"]:
            autodl_config["data"]["local_image_dir"] = autodl_config["data"]["autodl_image_dir"]

    # 更新量化模型路径
    if "quantized_model" in autodl_config and "autodl_dir" in autodl_config["quantized_model"]:
        autodl_config["quantized_model"]["local_dir"] = autodl_config["quantized_model"]["autodl_dir"]

    # 更新合并模型路径
    if "merged_model" in autodl_config and "autodl_dir" in autodl_config["merged_model"]:
        autodl_config["merged_model"]["local_dir"] = autodl_config["merged_model"]["autodl_dir"]

    # 更新输出路径
    if "output" in autodl_config and "autodl_dir" in autodl_config["output"]:
        autodl_config["output"]["local_dir"] = autodl_config["output"]["autodl_dir"]

    # 更新环境标识
    if "environment" in autodl_config:
        autodl_config["environment"]["current"] = "autodl"

    return autodl_config


def generate_local_config(config: dict) -> dict:
    """生成本地配置"""
    local_config = config.copy()

    # 更新环境标识
    if "environment" in local_config:
        local_config["environment"]["current"] = "local"

    return local_config


def save_config(config: dict, output_path: str):
    """保存配置文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser(description="生成量化评测配置")
    parser.add_argument("--config", required=True, help="输入配置文件")
    parser.add_argument("--environment", default="auto", choices=["local", "autodl", "auto"], help="目标环境")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有文件")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 确定环境
    if args.environment == "auto":
        # 自动检测环境
        if os.path.exists("/root/autodl-tmp"):
            environment = "autodl"
        else:
            environment = "local"
    else:
        environment = args.environment

    print(f"目标环境: {environment}")

    # 生成配置
    if environment == "autodl":
        new_config = generate_autodl_config(config)
    else:
        new_config = generate_local_config(config)

    # 确定输出路径
    if args.output_dir:
        output_dir = args.output_dir
    elif environment == "autodl" and "output" in config and "autodl_dir" in config["output"]:
        output_dir = os.path.dirname(config["output"]["autodl_dir"])
    else:
        output_dir = "configs/quantization"

    # 生成输出文件名
    config_name = os.path.basename(args.config).replace(".yaml", "")
    output_filename = f"autodl_{config_name}.yaml" if environment == "autodl" else f"local_{config_name}.yaml"
    output_path = os.path.join(output_dir, output_filename)

    # 检查是否已存在
    if os.path.exists(output_path) and not args.overwrite:
        print(f"文件已存在: {output_path}")
        print("使用 --overwrite 覆盖")
        return

    # 保存配置
    save_config(new_config, output_path)
    print(f"配置已保存: {output_path}")

    # 打印关键配置
    print("\n关键配置:")
    if "model" in new_config:
        print(f"  模型路径: {new_config['model'].get('local_dir', 'N/A')}")
    if "data" in new_config:
        print(f"  数据路径: {new_config['data'].get('local_metadata_dir', 'N/A')}")
    if "output" in new_config:
        print(f"  输出路径: {new_config['output'].get('local_dir', 'N/A')}")


if __name__ == "__main__":
    main()
