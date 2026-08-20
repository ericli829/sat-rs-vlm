# Balanced Probe Dataset

## 根因

旧 `cyclic_full_coverage` 对每个 task 使用固定 bucket size。以 1222 条 Scene、bucket 1200
为例，各轮为 `1200, 22, 0, ...`。旧 ViT probe 又只读取一个 round，并在 task quota 不足时
由同 source 的其他 task 静默补齐，因此配置中的 Scene 900 并不等于实际 Scene 900。

Stage-A v2 将完整、唯一、排除评测集的 canonical population 作为唯一新实验抽样源：

```text
raw VRSBench / LEVIR-CC
  -> normalization + formal prompt/path rewrite
  -> exclude Unified E1/E2/E3 IDs
  -> population_manifest.json
  -> balanced probe
```

历史 `build_probe_dataset()` 和旧 probe 文件不改变，继续用于历史复现。新实验使用
`sat_rs_vlm.data.probe_sampling.build_balanced_probe_dataset()` 与通用 CLI。

## 严格配额

默认 `quota_shortfall_policy: error`。任意 source/task 不足都会报告 requested、available 和
shortfall 并终止。只有显式设置 `redistribute` 才可补齐总量；此时 manifest 仍记录原始
shortfall，并保持 `quota_satisfied=false`，不能把补齐结果解释为满足原 task 配额。

输出目录包含：

```text
train.jsonl
manifest.json
distribution_report.md
```

Manifest 记录 requested/available/selected distribution、重复策略、全部 sample IDs、保护集
交集、canonical manifest SHA 和输出 SHA。`--dry-run` 只返回统计，不写训练资产。

## 命令

```bash
python scripts/data/build_balanced_probe_dataset.py \
  --config configs/data/qwen3vl_4b_balanced_probe.yaml \
  --dry-run

python scripts/data/build_balanced_probe_dataset.py \
  --config configs/data/qwen3vl_4b_balanced_probe.yaml
```

未来若仍需 cyclic 数据，使用 `balanced_cyclic_full_coverage`。它把每个 task 均匀切成
`num_rounds` 份，保证每 task 的轮间数量差不超过 1；旧 `cyclic_full_coverage` 保持固定桶
语义，仅用于历史复现。
