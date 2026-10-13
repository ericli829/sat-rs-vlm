# DINO + GeoRSCLIP 两阶段筛选

## 流程

1. 用分支里的 DINO proposal provider 按大类同义词检测，并执行 tiled 合并、NMS 和已有实例细化。
2. 对每个 DINO 框，用固定大类正词原型减固定干扰负词原型。分差大于粗筛阈值的框进入“无限定词结果”。车辆组沿用 `7 × 3 + 8 × 3 = 45` 条提示词；其余类别组见 `configs/prompts/georsclip_major_categories.yaml`。负词固定在配置中，不按图像抽样。
3. 只对粗筛保留的框评分限定词。限定词分差为：

   `0.80 × (限定目标 − 无限定目标) + 0.20 × (限定目标 − 对照限定目标)`

   综合排序分为：

   `0.50 × 大类初筛分差 + 0.50 × 限定词分差`

   最终候选要求大类初筛分差和限定词分差都严格大于各自阈值，默认阈值均为 `0`。入选框按综合排序分从高到低排列；权重与阈值可由配置覆盖。

   例如上游传入 `target_category="car"`、`modifier="red"` 时，默认三条查询是 `red car`、`car`、`non-red car`。限定词和目标类别都来自上游输入，不在算法中固定成某一组。限定词查询是直接短语，不添加额外模板。

GeoRSCLIP 对候选框使用原始框裁剪后灰色补边为正方形，再送入模型。这样不让 OpenCLIP 的中心裁切截掉细长的小目标，也与此前候选框打分的预处理一致。此策略由 `region_crop_policy` 控制。

CLIP 分数是余弦相似度与分差，不是概率。JSON 和 CSV 保留全部 DINO 框与阶段分数；图片分别展示大类粗筛结果和限定词结果。

## 配置与运行

先设置 `configs/integrations/dino_georsclip_two_stage.example.yaml` 中引用的 LAE-DINO 和 GeoRSCLIP 环境变量，再运行：

```powershell
python scripts/integrations/run_dino_georsclip_two_stage.py `
  --image path/to/image.jpg `
  --question "How many red cars are there?"
```

对象与修饰词可由上游分别传入；例如：

```powershell
python scripts/integrations/run_dino_georsclip_two_stage.py `
  --image path/to/image.jpg `
  --target car `
  --modifier red `
  --contrast-modifier non-red
```

Python 调用时也可以直接传入结构化目标；系统会按提示词表自动确定大类提示词组：

```python
result = strategy.run(
    image_path,
    target_category=upstream_target,
    modifier=upstream_modifier,
)
```

如果已经有同一张图的 DINO 结果，可用 `--proposal-json` 重放框并只重算 GeoRSCLIP 两阶段分数。输出 JSON 会标明框来自缓存，DINO 未重新运行：

```powershell
python scripts/integrations/run_dino_georsclip_two_stage.py `
  --image path/to/image.jpg `
  --target car `
  --modifier red `
  --proposal-json path/to/dino_results.json
```

输出目录包含：

- `all_dino_candidates.csv`、`all_dino_candidates_ranked.png`：DINO 全候选及大类分差排序；
- `unqualified_results.csv`、`unqualified_results.png`、`unqualified_source_overlay.png`：大类粗筛结果；
- `qualified_results.csv`、`qualified_results.png`、`qualified_source_overlay.png`：通过两级正分差条件的限定词结果，按大类分差与 80/20 限定词分差的 50/50 综合分排序；
- `two_stage_results.json`：检测器信息、实际提示词、全部逐框分数和两个阶段的结果。
