# Bit Flip 可靠性工具迁移计划

## 审计基线

- 最终项目：当前 `sat-rs-vlm` 工作区。
- 只读参考：同级 `sat-rs-vlm-temp` 参考仓库。
- 主项目分支：`server-adaption`，审计时 HEAD 为
  `b7b938f2ec5e3342859dc6249e176c1d62f7cfe6`，工作区状态为空。
- 来源提交：`e896b9113b781315adbaadd58c18549610d74c9f`，提交说明为
  `Add single-event-upset reliability toolkit`。
- 来源提交以主项目当前 HEAD 为父提交，共新增 28 个文件；参考目录只读，迁移不采用目录覆盖、
  跨目录合并或分支切换。
- 主项目已有分层 YAML、环境变量展开、统一路径、数据集 manifest、Qwen3-VL/LoRA 评测加载、
  JSONL 工具和实验环境记录。可靠性功能必须复用这些边界。

## 逐文件迁移映射

| 来源文件 | 原功能 | 主项目中已有类似功能 | 处理方式 | 最终位置 | 说明 |
|---|---|---|---|---|---|
| `configs/reliability/bitflip_lora_smoke.yaml` | LoRA bit flip smoke 参数 | 有，分层配置与本地/云端路径层 | 改为配置项 | `configs/reliability/base.yaml`、`local_smoke.yaml`、`experiments/lora_bitflip.yaml` | 移除本机绝对路径回退，区分 Mock 与真实模式 |
| `configs/reliability/protection_suite_smoke.yaml` | 四类保护策略 smoke 参数 | 有，统一 output root 与环境覆盖 | 与当前实现合并 | `configs/reliability/protection_suite.yaml`、`local_smoke.yaml` | 不保留独立配置加载器及仓库内固定报告路径 |
| `scripts/build_checksum_manifest.py` | 构建目录 SHA-256 manifest | 有，单文件流式 SHA-256 | 改为统一脚本入口 | `scripts/reliability/checksum_manifest.py build` | 核心移入 `checksum.py`，仅存相对路径 |
| `scripts/build_eval_sample_manifest.py` | 按任务固定抽样 | 有，DatasetManifest 与 split 读取 | 只迁移核心算法 | `src/sat_rs_vlm/data/reliability_manifest.py`、`scripts/data/build_reliability_eval_manifest.py` | 增加 split 泄漏检查，不写绝对图片路径 |
| `scripts/compare_adapter_fault_outputs.py` | clean/fault 配对与指标 | 有，评估结果 JSONL | 改为应用服务 | `evaluation/reliability/metrics.py`、`application/reliability_service.py` | 删除 `adapter_aware_mock_predict`；Mock 数据必须显式标记 |
| `scripts/evaluate_fault_tolerance.py` | 批量故障级别实验 | 有，Qwen3-VL 评测入口 | 改为应用服务 | `application/reliability_service.py`、`scripts/reliability/run_experiment.py` | 真实模式通过现有评估调用接口执行，不复制模型加载 |
| `scripts/inject_adapter_bitflip.py` | safetensors Adapter 注入 | 无完整实现 | 只迁移核心算法 | `models/reliability/fault_injector.py` | 保留 metadata、复制配置、校验重载与 clean hash |
| `scripts/plot_reliability_results.py` | 绘制多个旧报告 | 无 | 改为统一脚本入口 | `evaluation/reliability/plotting.py`、`scripts/reliability/plot_results.py` | 只读标准 metrics，不注入或推理 |
| `scripts/run_adapter_file_fault_smoke.py` | 小型 Adapter 文件 smoke | 无 | 改为统一脚本入口 | `scripts/reliability/run_smoke.py --case adapter-file` | fake adapter 只用于 `smoke_mock` |
| `scripts/run_adapter_recovery_smoke.py` | 文件恢复 smoke | 无 | 改为统一脚本入口 | `scripts/reliability/run_smoke.py --case recovery` | 使用原子替换和恢复后校验 |
| `scripts/run_output_guard_smoke.py` | 输出过滤和投票 smoke | 无 | 改为统一脚本入口 | `scripts/reliability/run_smoke.py --case output-guard` | 复用统一验证器和投票策略 |
| `scripts/run_protection_suite.py` | 四类保护策略编排 | 无 | 改为应用服务 | `models/reliability/protection.py`、`application/reliability_service.py` | 删除脚本内 YAML、路径、JSONL 和业务逻辑 |
| `scripts/run_state_dict_fault_smoke.py` | fake state dict smoke | 无 | 改为统一脚本入口 | `scripts/reliability/run_smoke.py --case state-dict` | CPU 小张量验证选择器与可复现记录 |
| `scripts/run_tensor_bitflip_smoke.py` | tensor 单/多 bit smoke | 有 bytes/int 基础 API | 改为统一脚本入口 | `scripts/reliability/run_smoke.py --case tensor` | tensor 算法并入唯一 `bitflip.py` |
| `scripts/run_weight_clamp_smoke.py` | 权重范围裁剪 smoke | 无 | 改为统一脚本入口 | `scripts/reliability/run_smoke.py --case weight-clamp` | 明确标记实验性、依赖干净参考 |
| `scripts/verify_checksum_manifest.py` | 验证 checksum manifest | 有，单文件流式 SHA-256 | 改为统一脚本入口 | `scripts/reliability/checksum_manifest.py verify` | 与 build 共用结构化核心，并检查大小 |
| `src/sat_rs_vlm/models/reliability/__init__.py` | 可靠性包导出 | 目录存在但无包导出文件 | 与当前实现合并 | 同路径 | 只导出稳定公共 API，避免导入可选模型依赖 |
| `src/sat_rs_vlm/models/reliability/model_fault_injector.py` | state dict 随机注入和差异统计 | 无 | 合并并重写 | `models/reliability/fault_injector.py` | 增加正则、层、LoRA A/B、Adapter 目录注入 |
| `src/sat_rs_vlm/models/reliability/output_validator.py` | detection/counting 基础验证 | 有统一任务枚举/输出 schema | 只迁移核心算法 | 同路径 | 扩展 JSON、NaN/Inf、VQA、稳定错误码与规范化输出 |
| `src/sat_rs_vlm/models/reliability/recovery.py` | checksum 恢复和文本投票 | 无 | 合并并重写 | 同路径 | 投票放到 protection；恢复采用临时文件和原子替换 |
| `src/sat_rs_vlm/models/reliability/tensor_bitflip.py` | tensor 定点/随机翻转 | 有 bytes/int bit flip | 与当前实现合并 | `models/reliability/bitflip.py` | 建立唯一算法实现；旧 `bitflip_simulator.py` 仅兼容导出 |
| `src/sat_rs_vlm/models/reliability/weight_protection.py` | clean 范围 clamp | 无 | 与当前实现合并 | `models/reliability/protection.py` | 不建立单独碎片模块，返回结构化统计 |
| `tests/unit/test_compare_adapter_fault_outputs.py` | clean/fault 指标测试 | 有评估测试组织 | 仅保留测试思路 | `tests/unit/reliability/test_metrics.py` | 测试统一报告 schema 和按任务统计 |
| `tests/unit/test_model_fault_injector.py` | state dict 过滤和变化测试 | 无 | 与当前实现合并 | `tests/unit/reliability/test_fault_injector.py` | 补 LoRA A/B、正则、clean 不变和 safetensors |
| `tests/unit/test_output_validator.py` | detection/counting 测试 | 有任务类型定义 | 与当前实现合并 | `tests/unit/reliability/test_output_validator.py` | 补空值、非有限值、VQA 和稳定错误码 |
| `tests/unit/test_protection_suite.py` | 旧脚本参数与表格测试 | 有分层配置 | 仅保留测试思路 | `tests/unit/reliability/test_protection.py`、`tests/integration/reliability/test_report_pipeline.py` | 不再测试脚本内部业务函数 |
| `tests/unit/test_recovery.py` | 文本投票测试 | 无 | 与当前实现合并 | `tests/unit/reliability/test_recovery.py`、`test_protection.py` | 分开验证原子文件恢复与输出投票 |
| `tests/unit/test_tensor_bitflip.py` | tensor 可复现、索引检查 | 有 bytes/int 测试 | 与当前实现合并 | `tests/unit/reliability/test_bitflip.py` | 覆盖多 bit、不原地修改和结构化记录 |

## 最终边界

1. `models/reliability` 只负责可复用的故障、校验、保护与恢复算法。
2. `application/reliability_service.py` 负责一次实验的状态流转和标准目录。
3. `evaluation/reliability` 只负责指标、报告与绘图数据。
4. `data/reliability_manifest.py` 复用当前 DatasetManifest 读取固定评测集。
5. `scripts/reliability` 与 `scripts/data` 只解析参数并调用 `src` API。
6. `smoke_mock` 与 `real_inference` 使用不同显式入口状态；真实模式缺少依赖、GPU、模型、数据或
   Adapter 时立即失败，绝不自动回退。
7. 兼容模块 `bitflip_simulator.py` 仅重新导出 `flip_random_bit` 和 `flip_bit_at`，不保留第二份算法。

## 不迁移内容

- 所有脚本中的 `sys.path.insert`、独立 YAML/JSONL/path helper。
- 旧配置中的 Windows 绝对路径和固定 `reports/...` 输出。
- `adapter_aware_mock_predict` 以及任何从真实模式静默降级到 Mock 的逻辑。
- 15 个平铺脚本的文件形态；其能力收敛为 5 个薄入口。
- 直接覆盖部署 Adapter 的恢复实现；替换为临时文件校验后原子替换。
