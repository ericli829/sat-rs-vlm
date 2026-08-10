"""
量化模型评估脚本

专门用于评估 INT8 量化后的模型，与量化前的 LoRA 模型进行对比。
使用 VRSBench 数据集。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_vrsbench_data(data_dir: str, val_type: str = "vqa") -> list:
    """加载 VRSBench 数据"""
    if val_type == "vqa":
        with open(os.path.join(data_dir, "VRSBench_EVAL_vqa.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    elif val_type == "cap":
        with open(os.path.join(data_dir, "VRSBench_EVAL_Cap.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    elif val_type == "referring":
        with open(os.path.join(data_dir, "VRSBench_EVAL_referring.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def keyword_hit(prediction: str, reference: str) -> bool:
    """计算简单关键词命中"""
    import re
    tokens = set(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", reference.lower()))
    if not tokens:
        return False
    return any(token in prediction.lower() for token in tokens)


def valid_json(text: str) -> bool:
    """判断文本是否是合法 JSON"""
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def load_quantized_model(model_path: str):
    """加载量化模型"""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    import torch.quantization as quantization

    print("  加载基座模型...")
    base_model_path = "D:/Models/Qwen3-VL-2B-Instruct"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("  应用动态量化...")
    # 量化 Linear 层（仅包含 Linear 层，跳过 Embedding 层）
    quantized_model = quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    print(f"  加载量化权重: {model_path}/model.pt")
    state_dict = torch.load(os.path.join(model_path, "model.pt"), weights_only=True)
    quantized_model.load_state_dict(state_dict)
    quantized_model.eval()

    return quantized_model


def evaluate_model(model, processor, val_data: list, image_dir: str,
                   num_samples: int = 20, task_type: str = "vqa") -> dict:
    """评估模型"""
    from PIL import Image

    predictions = []
    for item in val_data[:num_samples]:
        try:
            image_id = item.get("image_id")
            if not image_id:
                continue

            image_path = os.path.join(image_dir, image_id)
            if not os.path.exists(image_path):
                continue

            # 根据任务类型构建问题
            if task_type == "vqa":
                question = item.get("question", "")
                reference = item.get("ground_truth", "")
            elif task_type == "captioning":
                question = "Please describe this image in detail."
                reference = item.get("caption", "")
            elif task_type == "detection":
                question = f"What objects are in this image? Please output JSON format with bbox and label."
                reference = json.dumps(item.get("objects", []))
            else:
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
                "task_type": task_type,
                "prediction": pred_answer,
                "reference": reference,
                "latency_ms": latency,
            })

            idx = len(predictions)
            print(f"  [{idx}/{num_samples}] {task_type}: {latency:.0f}ms")

        except Exception as e:
            print(f"  警告: {e}")
            continue

    # 计算指标
    if not predictions:
        return {"overall": {"num_samples": 0}, "by_task": {}}, []

    summary = {
        "overall": {
            "num_samples": len(predictions),
            "empty_predictions": sum(1 for p in predictions if not p["prediction"].strip()),
            "inference_latency_ms": np.mean([p["latency_ms"] for p in predictions]),
        },
        "by_task": {
            task_type: {
                "num_samples": len(predictions),
                "keyword_hit_rate": sum(keyword_hit(p["prediction"], p["reference"]) for p in predictions) / len(predictions),
                "average_generation_length": np.mean([len(p["prediction"]) for p in predictions]),
                "empty_prediction_rate": sum(1 for p in predictions if not p["prediction"].strip()) / len(predictions),
            }
        }
    }

    if task_type == "detection":
        summary["by_task"][task_type]["valid_json_rate"] = sum(valid_json(p["prediction"]) for p in predictions) / len(predictions)
    if task_type == "vqa":
        summary["by_task"][task_type]["exact_match_rate"] = sum(
            p["prediction"].strip().lower() == p["reference"].strip().lower()
            for p in predictions
        ) / len(predictions)

    return summary, predictions


def main():
    parser = argparse.ArgumentParser(description="评估量化模型")
    parser.add_argument("--model-path", required=True, help="模型路径")
    parser.add_argument("--is-quantized", action="store_true", help="是否为量化模型")
    parser.add_argument("--data-dir", required=True, help="VRSBench数据目录")
    parser.add_argument("--image-dir", required=True, help="图片目录")
    parser.add_argument("--num-samples", type=int, default=20, help="评估样本数")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--model-label", default="model", help="模型标签")
    parser.add_argument("--task-type", default="vqa", choices=["vqa", "captioning", "detection"], help="任务类型")
    args = parser.parse_args()

    print(f"模型路径: {args.model_path}")
    print(f"是否量化: {args.is_quantized}")
    print(f"模型标签: {args.model_label}")
    print(f"任务类型: {args.task_type}")

    # 加载验证数据
    val_data = load_vrsbench_data(args.data_dir, args.task_type)
    print(f"验证集样本数: {len(val_data)}")

    # 加载模型
    print("\n加载模型...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    if args.is_quantized:
        model = load_quantized_model(args.model_path)
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # 评估
    print(f"\n评估 {args.num_samples} 条样本...")
    summary, predictions = evaluate_model(
        model, processor, val_data, args.image_dir, args.num_samples, args.task_type
    )

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    summary_file = os.path.join(args.output_dir, f"{args.model_label}_{args.task_type}_summary.json")
    predictions_file = os.path.join(args.output_dir, f"{args.model_label}_{args.task_type}_predictions.jsonl")

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(predictions_file, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n保存结果:")
    print(f"  摘要: {summary_file}")
    print(f"  预测: {predictions_file}")

    return summary


if __name__ == "__main__":
    main()
