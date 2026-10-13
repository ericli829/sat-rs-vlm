# Planner 稳定版本核验

状态：**未能唯一确认 2026-10-10 的生产稳定版；生产 LoRA 环境变量保持原样，未指向候选版本。**

本地 `D:\Desktop\tzb-2026\final models` 只发现两个 2026-09-02 的 Planner 目录。云端已检查 `/root/autodl-tmp/outputs/taskgraph`、`/root/autodl-fs/data/sat-rs-vlm-training` 与 `/root/autodl-fs/experiments` 中 2026-10-09 至 2026-10-11 创建的 adapter 配置；找到 10 月 11 日清洗训练产物，未找到 10 月 10 日的 adapter。当前环境没有设置 `QWEN3VL_4B_PLANNER_LORA`，仓库生产示例仍由部署环境注入该变量。

| 候选 | 证据 | Base / LoRA | adapter 权重 SHA256 | 结论 |
|---|---|---|---|---|
| `D:\Desktop\tzb-2026\final models\qwen3vl_4b_planner_hard_refine_v1_20260902_234235` | 云端目录时间 2026-09-03；训练与复用记录见 `reports/training/planner_changechat_plan_20261011.md` | Qwen3-VL-4B-Instruct；r=16、alpha=32、dropout=0.05、7 个语言投影模块 | `defad37c62a53cb928d8879d8f690cc55806dffee7a81a547f121c9975511570` | 可确认的早期 hard-refine adapter，不能证明它就是 10 月 10 日冻结版本 |
| `/root/autodl-fs/data/sat-rs-vlm-training/planner_4b_clean_20261011/output/training`（本地副本 `reports/training/planner_4b_clean_20261011/cloud_download`） | 10 月 11 日训练目录；评测报告记载该 checkpoint 的 validation runtime-valid 197/200，final adapter 与 checkpoint-351 相同 | 同一 Qwen3-VL-4B-Instruct；r=16、alpha=32、dropout=0.05、相同 7 个模块 | `d7169cce768dcd01e86f1c2fd2847f7fddf85e45f9d61db3e23180a97211a4e6` | 日期晚于要求，且不等于所需 10 月 10 日版本 |

`reports/training/planner_final_mix_20261013/FINAL_RESULTS.md` 将一个基线称为“稳定 Planner LoRA”，但它记录的是评测结果，没有给出可追溯到 10 月 10 日的路径和权重 SHA256；因此不能据此覆盖生产指针。两个可见候选的 base、rank、alpha、dropout 和目标模块一致，但 Processor/Base 文件哈希与 10 月 10 日版本血缘仍没有证据。

下一步如需解除此阻塞，应从 10 月 10 日部署记录或当日模型归档中取得精确 adapter 路径/文件，再核对权重 SHA256 和 Base/Processor。这里没有启动训练，也没有变更任何模型路径。
