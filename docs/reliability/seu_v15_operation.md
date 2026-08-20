# SEU v1.5 操作说明

默认工程是 `lyz-reliability-v15-migration` 分支。默认配置是全覆盖敏感性扫描，不是 LoRA-only 扫描。

## 本地预检

```bash
python scripts/reliability/run_v15_sensitivity.py --environment autodl --preflight
```

本机没有 CUDA 时，预检可以完成路径和依赖检查，但真实推理会明确报告 CUDA 不可用，不会降级到 mock。

## 查看完整计划

```bash
python scripts/reliability/run_v15_sensitivity.py --environment autodl --run-id v15-full-scan --dry-run
```

默认计划覆盖 language model、Attention、MLP、vision encoder、embeddings、LoRA adapter；语言模型第 0 到 27 层；sign、exponent、mantissa；1、10、100 bit；每个条件 3 次重复。只有显式指定 `--targets` 时，才缩小为指定目标集合。

## AutoDL 完整扫描

```bash
python scripts/reliability/run_v15_sensitivity.py --environment autodl --run-id v15-full-scan
```

中断后使用同一个 run id 续跑：

```bash
python scripts/reliability/run_v15_sensitivity.py --environment autodl --run-id v15-full-scan --resume
```

已完成条件会跳过，失败或缺失条件单独补跑。

## GPU 单点验证

完整扫描前建议先验证 Attention 第 0 层、MLP 第 14 层和 LoRA 第 27 层各一个条件。具体命令见 `实施路线.md` 和本文件对应配置。

## 绘图

绘图只读取已有结果，不重新推理：

```bash
python scripts/reliability/plot_results.py --sensitivity-root <RUN_DIR> --output <RUN_DIR>/figures
```

输出包括扫描覆盖率、层 x bit-plane 热力图、bit-plane 对比和结构化汇总数据。

## 边界

- `real_inference` 必须有 CUDA、真实模型、Adapter 和数据集，缺少任一项直接失败。
- smoke/mock 只能验证流程，不能替代真实敏感性结论。
- KV Cache 和激活值不是参数扫描对象，需通过运行时故障和 Activation Guard 单独评估。
