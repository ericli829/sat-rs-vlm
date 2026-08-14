# 固定分层评测集 E1/E2/E3

项目把 VRSBench 与 LEVIR-CC 的合法 validation population 冻结为三个嵌套层级：

```text
E1 Quick (593) < E2 Standard (3000) < E3 Full (69577)
```

E1/E2 是提高有限评测预算信息密度的分层诊断集，不代表原始数据自然比例；E3 才是完整合法评测总体。所有文件均固定在 `data/evaluation/tiers/`，评测时不再运行随机抽样或 `max_eval_samples` 截断。

## 层级语义

| Tier | 用途 | 是否默认 |
|---|---|---|
| E1 Quick | debug 后检查、小型 A/B、loss 实验、checkpoint 初筛 | 否 |
| E2 Standard | 正式训练后评测、H1、ViT unfreeze、主要方案比较 | 是 |
| E3 Full | 最终候选、阶段结论、最终量化/可靠性和论文报告 | 否 |

现有量化实验 `contrast_20260811_191408` 的 593 个固定样本被完整保留为 E1。冻结资产不仅保留 ID，还恢复历史 sample manifest 的 task、prompt 和 reference；因此旧 baseline 仍能做严格 paired comparison。E2 从 E1 增补，E3 从 E2 增补，同一 ID 在三个层级中内容完全一致。

## 分层规则

采样 seed 固定为 42，分层优先级为 dataset/source、task_type、task subtype：

- Detection：归一化 bbox 面积桶 `small <= 0.01`、`medium <= 0.10`、`large > 0.10`，并结合 class label。
- Counting：`0`、`1`、`2`、`3`、`4`、`5-9`、`10+`；无法可靠转成整数的定性答案进入显式 `unresolved` 桶，不猜数值。
- VQA / Scene：只使用真实 `metadata.qa_type`，不根据问题文本推断。
- LEVIR-CC：按真实 `metadata.changeflag` 的 0/1 分层。

配额使用 sqrt-population 权重并优先覆盖现有 strata，避免高频类别完全主导 E1/E2，也不把诊断集强行解释成自然总体。

## 资产与审计

```text
data/evaluation/tiers/
├── e1_quick.jsonl
├── e2_standard.jsonl
├── e3_full.jsonl
├── legacy_fixed_593_ids.txt
├── legacy_fixed_593_samples.jsonl
└── evaluation_tiers_manifest.json
```

manifest 保存 source SHA256、population/tier distribution、sampling fraction、所有 tier ID、bbox/count 阈值、非法样本排除原因和子集/泄漏不变量。当前 VRSBench 原始 validation 中有 6 条零宽或零高 bbox，被明确排除于合法总体；未修改或猜测其框。

重新生成：

```powershell
$env:LEVIR_CC_ROOT = "E:\迅雷下载\LEVIR-CC"
python scripts/evaluation/build_evaluation_tiers.py `
  --config configs/eval/evaluation_tiers.yaml
```

AutoDL 中把 `LEVIR_CC_ROOT` 指向数据集根目录即可。生成器会检查 source checksum、ID 唯一性、train/eval 交集、历史 E1 完整性和 E1/E2/E3 的严格包含关系。

## 运行评测

默认 E2：

```bash
python scripts/evaluate_rs_vlm.py \
  --config configs/eval/qwen3vl_eval.yaml \
  --checkpoint /path/to/experiment
```

显式 E1 或 E3：

```bash
python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval.yaml \
  --checkpoint /path/to/experiment --eval-tier E1
python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval.yaml \
  --checkpoint /path/to/experiment --eval-tier E3
```

也可以直接选用 `qwen3vl_eval_e1.yaml`、`qwen3vl_eval_e2.yaml` 或 `qwen3vl_eval_e3.yaml`。三个配置都设置 `max_eval_samples: null`。

评测 manifest 写入 `evaluation_tier` 和 `evaluation_tier_sha256`。paired comparison 在双方 tier 或 SHA256 不一致时直接失败；旧 manifest 缺少 tier 元数据时保留兼容警告，并继续执行严格 ID、task 和 reference 校验。

## 防止训练泄漏

`hard_example_mining.load_evaluation_ids()` 能直接读取 `evaluation_tiers_manifest.json` 的 E3 ID 集。H1 默认配置使用该 manifest，因此 E1/E2/E3 的全部评测样本都不会进入 hard examples、regular replay 或 H1 training。
