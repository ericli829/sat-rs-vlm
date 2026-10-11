"""Generate independent, reproducible Planner localization SFT (no model)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from taskgraph_lab.synthetic_locate import (
    FAMILIES,
    ROOT,
    VERSION,
    json_text,
    make_samples,
    quality_statistics,
    sft_row,
    sha256_file,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json_text(row) + "\n")


def quality_report(stats: dict) -> str:
    return "\n".join(
        [
            "# 程序化定位训练数据质量报告",
            "",
            f"实际生成 {stats['total']} 条，划分 {stats['split_counts']}。"
            f"一节点 LOCATE 共 {stats['one_node_locate_count']} 条，"
            f"占 {stats['one_node_locate_fraction']:.1%}。",
            "所有问题为英文。无图像、无真实答案框；仅监督规划路径。"
            "唯一性属于 synthetic planning assumption，Executor 必须拒绝零目标、多目标和未决证据。",
            "",
            "## 静态检查",
            "",
            "逐条使用当前 Schema、Validator、TypeChecker、compiler/parser 规范化往返及约束语法"
            "完整接受检查；独立校验结构化参数、自然语言、最短算子路径与生产输入。",
            "每字符前缀在独立验证命令的 --every-prefix 模式检查；"
            "不能把完整接受检查单独称为已逐字符穷举。",
            "~~~json",
            json.dumps(stats, ensure_ascii=False, indent=2),
            "~~~",
            "",
            "## 划分与重复",
            "",
            "按类别、动作句式家族以及可分离的空间规则值联合隔离。训练使用祈使句，验证使用独立疑问句，类别互斥。",
            "同一 split 内有意复用模板（统计见 within_split_reused_skeleton_rows）。"
            "跨 split 广义语言模板骨架、类别及原文交叉均应为 0。",
            "LOCATE→BBOX 等核心图骨架在训练和验证都出现，这是专项任务而非未见算子组合的"
            "泛化评测；不能声称结构完全独立。验证同时留出类别、句式和空间规则，不能单维归因；"
            "vehicle/car 等父子类别仍可能分属两侧。",
            "",
            "## 输入与训练",
            "",
            "每条 user 消息由生产 CompetitionRequestAdapter.adapt 和"
            " Qwen3VLPlannerProvider._messages 直接构造：FREE_FORM、choices=null、"
            "符号 image0 引用，以及当前真实 bbox 定位后缀。",
            "直接导出 system/user/assistant messages，assistant 仅规范 DSL。"
            "不经过 accepted 教师通道，不伪造 answer_audit。",
            "512 输出 token 与 1088 序列预算、真实训练 Dataset/collator 的 assistant-only loss "
            "检查记录在 sft_encoding_verification.json；"
            "没有该文件时不能宣称真实 tokenizer 已验证。",
            "",
            "## 原数据保护和限制",
            "",
            "现有 SFT、hard curriculum、固定 held-out test 和训练配置均不覆盖。"
            "旧 80 条问题只用于精确排重，不充当独立新评估集。",
            "mix_planner_synthetic 支持独立构建默认 20% 增量曝光混合集，"
            "保留原训练行和固定 test 字节。未启动训练或真实模型推理。",
            "",
        ]
    )


def manual_review(rows: list[dict]) -> str:
    # Cover every category first, then all families/splits, at least 40 rows.
    chosen = []
    covered = set()
    for row in rows:
        key = row["generation"]["category"]
        if key not in covered:
            chosen.append(row)
            covered.add(key)
    for split in ("train", "validation"):
        for family in FAMILIES:
            if not any(
                r["generation"]["split"] == split and r["generation"]["family"] == family
                for r in chosen
            ):
                candidate = next(
                    (
                        r
                        for r in rows
                        if r["generation"]["split"] == split and r["generation"]["family"] == family
                    ),
                    None,
                )
                if candidate:
                    chosen.append(candidate)
    text = [
        "# 人工审核样本",
        "",
        f"共 {len(chosen)} 条；以下为完整问题、实际 Planner 输入、Gold DSL 和结构化生成规则。"
        "没有审核真实视觉答案。",
        "",
        "所有样本实际使用的共同 system 消息（下方逐条完整列出 user 和 assistant）：",
        "~~~text",
        rows[0]["messages"][0]["content"],
        "~~~",
        "",
    ]
    for row in chosen:
        text.extend(
            [
                f"## {row['id']}",
                "",
                row["question"],
                "",
                "Planner 输入：",
                "~~~json",
                row["messages"][1]["content"],
                "~~~",
                "",
                "Gold DSL：",
                "~~~text",
                row["dsl"],
                "~~~",
                "",
                "生成规则：",
                "~~~json",
                json.dumps(row["generation"], ensure_ascii=False, indent=2),
                "~~~",
                "",
            ]
        )
    return "\n".join(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "taskgraph_lab/configs/planner_synthetic_locate_v1.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--ratios", help="JSON object of category/intrinsic/region/extreme/ordinal/semantic weights"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.count is not None:
        config["count"] = args.count
    if args.seed is not None:
        config["seed"] = args.seed
    if args.ratios:
        config["ratios"] = json.loads(args.ratios)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    old_questions = (
        ROOT / "taskgraph_lab/data/questions/semantic_locate_v1_20261010/questions.jsonl"
    )
    excluded = (
        {
            json.loads(line)["question"].strip().casefold()
            for line in old_questions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if old_questions.exists()
        else set()
    )
    rows, groups = make_samples(config, excluded_questions=excluded)
    stats = quality_statistics(rows)
    assert (
        stats["exact_text_duplicates"]
        == stats["cross_split_exact_text"]
        == stats["cross_split_categories"]
        == stats["cross_split_template_families"]
        == stats["cross_split_near_template_skeletons"]
        == 0
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "generator_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    write_jsonl(output / "synthetic_samples.jsonl", rows)
    for split in ("train", "validation"):
        write_jsonl(
            output / f"{split}.jsonl",
            [sft_row(r) for r in rows if r["generation"]["split"] == split],
        )
    manifest = {
        "generator_version": VERSION,
        "source": "synthetic_programmatic_plan",
        "config": config,
        "population_count": len(rows),
        "target_format": "dsl",
        "planner_dsl_version": "taskgraph-v1.1-dsl-v1",
        "prompt_version": rows[0]["metadata"]["prompt_version"],
        "extension_version": "planner-bbox-final-v1",
        "split_categories": groups,
        "statistics": stats,
        "old_80_questions_excluded": len(excluded),
        "old_80_exact_overlap": 0,
        "old_80_sha256": sha256_file(old_questions) if old_questions.exists() else None,
        "source_file_sha256": rows[0]["metadata"]["contract_file_sha256"],
        "generator_sha256": sha256_file(ROOT / "taskgraph_lab/synthetic_locate.py"),
        "files": {
            name: {"sha256": sha256_file(output / name)}
            for name in ("synthetic_samples.jsonl", "train.jsonl", "validation.jsonl")
        },
        "visual_answer_verified": False,
        "model_inference_executed": False,
        "training_started": False,
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "quality_report.md").write_text(quality_report(stats), encoding="utf-8")
    (output / "manual_review.md").write_text(manual_review(rows), encoding="utf-8")
    audit = ROOT / "reports/taskgraph/planner_synthetic_locate_v1/protocol_contract_audit.md"
    if audit.exists():
        shutil.copyfile(audit, output / "protocol_contract_audit.md")
    print(
        json.dumps(
            {"output": str(output), "count": len(rows), "stats": stats},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
