# 外部微调插件

## 设计目标

外部插件用于隔离尚未纳入稳定主线的微调方法。主项目只提供版本化 API、公共模型/数据/Trainer 服务、显式发现、依赖检查和统一报告；策略负责自己的配置验证、模型改造、训练参数、保存方式和报告字段。

主项目正常导入、LoRA 训练和 pytest 都不会扫描插件目录。插件根目录只能通过 `--plugin-root`、`SAT_RS_VLM_PLUGIN_ROOTS` 或 `configs/external_plugins.yaml` 显式给出。

## 本地目录

```text
.local_plugins/sat-rs-vlm-local-plugins/
├── plugin_pack.yaml
├── PLUGIN_SPEC.md
├── MIGRATION_REPORT.md
├── templates/strategy_plugin/
└── plugins/
    ├── qlora/
    ├── dora/
    ├── adalora/
    ├── ia3/
    ├── partial_unfreeze/
    ├── full_sft/
    └── prompt_tuning/
```

每个策略拥有独立 manifest、requirements、训练/冒烟配置、入口、文档、测试和输出目录。

## 命令

```powershell
$pluginRoot=".local_plugins/sat-rs-vlm-local-plugins"
python scripts/list_external_plugins.py --plugin-root $pluginRoot --validate
python scripts/validate_external_plugin.py --plugin-root $pluginRoot --strategy ia3
python scripts/run_external_strategy.py --plugin-root $pluginRoot --strategy ia3 --check-only
python scripts/run_external_strategy.py --plugin-root $pluginRoot --strategy ia3 --config "$pluginRoot/plugins/ia3/configs/smoke.yaml" --dry-run
```

真实 forward 或训练前设置 `LOCAL_MODEL_DIR`、`TRAIN_JSONL`、`VAL_JSONL` 和 `DATA_ROOT`。超过 30 秒的真实训练应按项目规则后台启动并记录 PID 与日志。

## 依赖策略

默认只报告 `satisfied`、`missing` 或 `conflict`。安装必须显式传入 `--install-missing`，并使用运行脚本的同一个 Python 解释器。已有包版本冲突时拒绝自动降级；离线模式只能从显式 wheel 目录安装。QLoRA 的 bitsandbytes 仅属于 QLoRA 插件。

## 安全与兼容

- manifest 和 API 主版本不一致时拒绝加载。
- 插件输出不得逃逸显式插件根目录。
- 插件入口使用唯一模块名加载，不永久修改 `sys.path`。
- 外部策略只允许导入 `sat_rs_vlm.plugins` 公开 API。
- 不支持、依赖缺失、资源不兼容和训练失败保持不同错误语义。
- 任一失败都不会回退到 LoRA。

## PluginContext 公共服务

插件通过 `context.service(name)` 获取白名单服务，不能导入主项目内部模块：

| 服务 | 作用 |
|---|---|
| `runtime_modules` | 延迟取得已检查的 Torch、Transformers、PEFT 和可选 bitsandbytes 模块 |
| `inspect_environment` | 返回库版本与 CPU/CUDA 设备摘要 |
| `load_base_model` / `load_processor` | 仅从显式本地目录加载 Qwen3-VL 和 Processor |
| `create_dataset` / `create_collator` | 复用稳定数据格式与 Qwen3-VL collator |
| `create_trainer` | 创建公共 Transformers Trainer |
| `forward_probe` | 对单 batch 做设备对齐后的前向兼容检查 |
| `parameter_summary` | 统计可训练参数、总参数和比例 |
| `match_module_suffixes` / `inspect_model_modules` | 验证目标模块并分类模型树 |
| `training_arguments_from_config` | 过滤公共 Trainer 支持的训练参数 |
| `save_adapter` / `save_full_model` | 分别保存 Adapter 或完整模型与 Processor |
| `write_json_report` / `resolve_path` | 写统一 JSON 报告和解析本地路径 |

服务集合是只读映射；未知服务名立即失败。重依赖只在显式插件命令真正调用相关服务时导入。

## 新增策略

复制本地 `templates/strategy_plugin/`，修改 manifest、策略类和两套配置，然后依次运行列表、验证、`--check-only` 和 `--dry-run`。真实模型测试不属于默认 pytest。
