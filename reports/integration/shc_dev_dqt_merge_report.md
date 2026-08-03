# SHC / dev-dqt 选择性整合与复审报告

## 1. 来源版本

- SHC：`d309e57d2c16d88b49e1f37c020811903e12b9f7`。
- dev-dqt：`40d2806ad7135aeaa213462adac7f5182ab3f2d1`，对应来源仓库的
  `origin/dev-dqt`。
- 整合前目标分支：`server-adaption`；整合前工作区状态详见
  `reports/integration/shc_dev_dqt_precheck.md`。
- 二次复审时目标分支：`master`，HEAD 为 `c06b3f9`。该提交已由用户在复审前创建；本次
  复审没有执行 commit 或 push。
- 只读来源 `sat-rs-vlm-temp` 复审时仍为干净工作区，两个 SHA 均可解析。

## 2. 合入内容

### SHC

- `data/qwen3vl_collator.py`：assistant-only loss mask、left/right padding、generation 无
  labels、截断后空监督检查。
- `data/task_protocol.py` 与 `prompt_templates.py`：counting/detection 统一 JSON、显式 bbox
  坐标格式、英文计数解析和 prompt 模板。
- `data/vrsbench.py` 与转换脚本：VRSBench 结构化 detection/counting、scene/VQA 正确映射、
  unresolved 样本降级。
- `data/sampling.py`、`task_sampler.py`：配额数据生成与可选 WeightedRandomSampler。
- `evaluation/schema.py`、`metrics.py`、`inference.py`：统一 prediction 字段、分任务指标和只
  解码新增 token 的生成流程。
- E0/E1/E1b/E1d 配置及 E1d-data、E1d-sampler、E1d-combined 的独立语义。

### dev-dqt

- `compression/quantization/`：`baseline`、`torch_dynamic_int8`、`bnb_int8` 三种后端。
- `scripts/quantize_rs_vlm.py`：统一量化入口；旧 `quantize_int8*.py` 仅保留为弃用 wrapper。
- 固定 messages 样本、Processor、adapter、seed、warmup 与 generation 的公平 benchmark。
- 延迟分位数、失败样本、任务指标、逻辑参数量、序列化字节和 CPU/CUDA 内存报告。
- CPU dynamic 产物明确标为 `benchmark_only`；bnb 保存后未 reload 前不声明可部署。

### 目标项目原有能力

- 保留分层 YAML、环境变量和本地/AutoDL 路径边界。
- 保留已跑通的默认 LoRA 命令、外部微调插件入口和 checkpoint loader。
- 保留 bit flip、checksum、恢复、output guard 和 weight clamp 可靠性模块。
- 量化、普通评估和可靠性评估复用共享 loader、messages、parser 和 task metrics，不合并各自
  的上层职责。

## 3. 未合入内容

- SHC 的完整 E1 派生 JSONL 和 dev-dqt 的缩减版 processed JSONL。
- 来源仓库中的本地评估报告、训练资产报告、checkpoint 和实验输出。
- 包含本机绝对路径的配置与命令。
- 两份重复量化大脚本、旧式 `instruction/image/answer` 读取路径和包含关系“准确率”。
- 未在当前硬件复现的加速比、精度保持率、部署可用性或可靠性结论。

## 4. 重复功能清理

| 重复能力 | 最终唯一边界 | 处理 |
| --- | --- | --- |
| Qwen3-VL 模型/Adapter 加载 | `models/qwen3vl_loader.py` | 普通评估和量化复用 |
| 生成与设备移动 | `evaluation/inference.py`、`training/utils.py` | 复用 `model_input_device`，只解码新 token |
| counting/detection parser | `data/task_protocol.py` | 指标和可靠性验证共享底层解析 |
| task metrics | `evaluation/metrics.py` | 普通评估、量化和可靠性指标复用 |
| messages 数据读取 | `Qwen3VLDataset`/`Qwen3VLDataCollator` | 量化不再扫描图片根目录或读取旧字段 |
| 量化配置/报告 | `compression/quantization/` | 两个旧脚本变为薄 wrapper |
| 环境变量展开 | `configuration/environment.py` | 新配置沿用 `MODEL_ROOT/DATA_ROOT/OUTPUT_ROOT` |

## 5. 已修复问题

- 修复 SHC 训练脚本 import 缩进错误并通过 compileall。
- assistant-only mask 覆盖左右 padding、图片 prompt、generation 和无 assistant token。
- counting 支持数字、英文数字、no/none 和 JSON；无法解析时不伪造数字。
- scene type 的 yes/no、存在性和属性问题保持 VQA。
- bbox 来源格式显式配置，不按数值范围猜测；百分比和像素坐标现在按声明正确验证。
- 单目标旧 `boxes/labels` 可转为新 `label/bbox`；多目标旧结构不静默截断。
- 无 bbox 的旧 detection 降级为 VQA，避免 JSON prompt 配自然语言监督答案。
- E1d 区分配额数据、加权 sampler 和组合实验；云端包装器完整转发相关字段。
- GPU INT8 正名为 `bnb_int8`，与 CPU `torch_dynamic_int8` 分离。
- 修复 `--skip-baseline`、空样本、缺图、多图、失败统计和 JSON-safe 报告。
- baseline 压缩元数据改为读取模型实际 device/dtype，不再固定写 CPU/float32。
- 默认 LoRA/QLoRA 最终 Adapter 现在生成 `strategy_manifest.json`，可被统一评估与可靠性入口
  识别。
- AutoDL 安装补齐 cloud requirements、`packaging` 和可选 `--install-qlora`；环境预检覆盖
  torchvision、safetensors、qwen-vl-utils 与 bitsandbytes。
- `.gitattributes` 强制 shell 和 Makefile 使用 LF，降低 Linux 云端换行风险。
- 修复 `.gitignore` 中未锚定的 `models/`：该规则曾错误忽略
  `src/sat_rs_vlm/models/` 下 8 个新增源码文件，导致本地文件存在但未进入提交，云端在
  pytest 收集阶段出现 12 个级联导入错误。规则现改为仅匹配仓库根目录的 `/models/` 和
  `/.models/` 模型缓存。
- 新增 `test_gitignore_integrity.py`，使用 `git check-ignore --no-index --stdin` 一次性检查
  `src/`、`scripts/` 和 `tests/` 下全部 Python 源码，防止已跟踪或未跟踪源码再次命中过宽
  的忽略规则。

## 6. 实际测试结果

复审前、整合代码提交后的完整测试：

```text
pytest -q
161 passed, 1 skipped, 1 warning in 39.31s
```

二次复审修改后的已完成验证：

```text
python -m compileall -q src scripts tests
通过

ruff check src tests scripts
All checks passed

ruff format --check src tests scripts
167 files already formatted

mypy src
Success: no issues found in 83 source files

pytest -q tests/unit
139 passed in 6.26s

目标协议、量化、manifest 与 AutoDL 包装测试
40 passed, 1 skipped in 11.42s

云端变量下全部 YAML 展开及训练/量化配置校验
5 passed in 3.17s

quantize_rs_vlm.py --dry-run
通过，2 个固定 fixture 样本

reliability/run_smoke.py --case all
通过，execution_mode=smoke_mock

二次复审后的完整 pytest
169 passed, 1 skipped, 1 warning in 26.16s

.gitignore 修复后的云端故障相关回归测试
52 passed（完整性门禁 1 项；可靠性、量化、评估与训练入口 51 项）
```

Windows 主机没有可用 Bash，因此 shell 的 `bash -n` 测试在 pytest 中标为 skipped；脚本
语法仍由 Linux 云端首次启动前复核。

## 7. 未完成与边界

- 真实 Qwen3-VL CPU dynamic INT8 尚未运行，需要足够主机内存；当前只验证 dry-run 与 toy
  Linear。
- `bnb_int8` 真实 benchmark、保存和 reload smoke 需要 CUDA 与 bitsandbytes。
- clean/fault/recovery 真实可靠性实验需要 CUDA、真实 LoRA Adapter、固定全量评测 manifest。
- AutoDL 真实 LoRA optimizer step、断点恢复、备份和 shell 执行仍需在服务器实际验证。
- 量化 base + fault adapter、量化恢复等组合尚未验证，不做支持声明。
- 本地 smoke 和模拟云配置只能证明工程调用链，不代表模型精度、速度、显存或抗辐射能力。

## 8. 最终 Git 状态

- 只修改目标仓库；只读来源仓库状态为空。
- 本次复审保留用户在 `master` 上的已有提交，只产生未提交的审计修复。
- `git diff --check` 无空白错误；Windows 仅提示现有工作树 LF/CRLF 转换，新增
  `.gitattributes` 已约束云端 shell 文件为 LF。
- 未新增完整训练数据、checkpoint 或本地真实性能报告。
- 本次复审未执行 reset、restore、checkout、clean、stash、merge、cherry-pick、rebase、
  commit 或 push。
