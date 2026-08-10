# 训练命令清单

以下命令均从项目根目录执行。Windows 多行续行符是反引号，Linux/macOS 使用反斜杠。

## 1. 本地创建环境

```bash
python scripts/environment/bootstrap_local.py --with-dev
```

创建 `.venv` 并安装基础、测试和静态检查依赖，不安装大型模型依赖。

## 2. 本地安装模型依赖

```bash
python scripts/environment/bootstrap_local.py --with-dev --with-model
```

复用或创建 `.venv`，增加 Transformers、PEFT 等真实模型训练依赖。

## 3. 激活本地环境

```powershell
.venv\Scripts\Activate.ps1
```

CMD 使用 `.venv\Scripts\activate.bat`，Linux/macOS 使用
`source .venv/bin/activate`。当前开发机验证例外地使用默认 Anaconda 环境。

## 4. 检查环境

```bash
python scripts/environment/check_environment.py
```

报告解释器、基础/模型依赖、CUDA、BF16、路径变量、输入目录、输出可写性和磁盘空间。

## 5. 导出环境报告

```bash
python scripts/environment/export_environment.py \
  --output reports/environment/local
```

生成 `environment.json`、`pip-freeze.txt`、`nvidia-smi.txt` 和 `command.txt`。

## 6. 校验数据集

```bash
python scripts/data/validate_dataset.py \
  --dataset-root /path/to/VRSBench \
  --manifest-name project_metadata/dataset_manifest.json
```

检查 manifest、分片、相对路径、图片、重复与交叉、任务字段和 bbox。

## 7. 创建本地 smoke 数据

```bash
python scripts/data/create_smoke_dataset.py \
  --source-root /path/to/VRSBench \
  --output-root data/samples/smoke \
  --sample-count 32
```

复制少量样本及其引用图片，生成可独立搬运的 embedded manifest。

## 8. 本地运行测试

```bash
pytest -q
```

默认跳过 `real_model`、`gpu`、`slow`、`cloud`，不联网也不读取完整数据集。

## 9. 本地真实模型测试

```bash
RUN_REAL_MODEL_TESTS=1 pytest -m real_model
```

PowerShell 先执行 `$env:RUN_REAL_MODEL_TESTS="1"`。该模式需要本地模型与足够资源。

## 10. 本地 LoRA Mock smoke

```bash
python scripts/training/run_smoke_train.py \
  --config configs/local/train_lora_smoke.yaml
```

只验证配置、manifest、输出、日志和 checkpoint 控制流，不判断模型效果。

## 11. 本地真实模型资产与前向验证

```powershell
$env:LOCAL_MODEL_DIR="<model-parent>\\Qwen3-VL-2B-Instruct"
$env:DATA_ROOT="<data-parent>\\VRSBench"
$env:TRAIN_JSONL="$PWD\\data\\processed\\qwen3vl_train.jsonl"
$env:VAL_JSONL="$PWD\\data\\processed\\qwen3vl_val.jsonl"

python scripts/validate_training_assets.py `
  --config configs/train/qwen3vl_local_smoke.yaml

python scripts/train_qwen3vl_lora.py `
  --config configs/train/qwen3vl_local_smoke.yaml `
  --forward-only
```

先检查模型、JSONL 与图片根目录，再只执行一个 batch 的真实 Qwen3-VL 前向传播。
这一步会加载本地模型，适合在正式训练前排查设备、处理器和样本格式问题。

## 12. 本地真实 LoRA 训练

```powershell
python scripts/train_qwen3vl_lora.py `
  --config configs/train/qwen3vl_local_smoke.yaml `
  --max-steps 2 `
  --output-dir checkpoints/qwen3vl-2b-vrsbench-lora/local-real-smoke
```

该命令执行真实 LoRA 的短 smoke 训练，验证反向传播、adapter 保存和 Trainer 状态。
正式训练改用 `configs/train/qwen3vl_local.yaml`，并通过 `--output-dir` 为每次实验提供
独立目录。真实训练需要可用 GPU；CPU 只应用于 Mock、配置与数据验证。

## 13. 本地真实推理

```powershell
python -m sat_rs_vlm.interfaces.cli infer `
  --config configs/local/qwen3vl_real_infer.yaml `
  --image "$env:DATA_ROOT\\<relative-image-path>" `
  --prompt "请描述这张遥感图像中的主要地物。"
```

该配置启用 HuggingFace 后端、只读取 `LOCAL_MODEL_DIR` 下的本地 Qwen3-VL 模型。可用
`--second-image <path>` 验证变化检测提示词；不要在真实模型测试中使用默认 mock 配置。

## 14. 外部插件策略检查与真实训练

```powershell
$pluginRoot="$PWD\\.local_plugins\\sat-rs-vlm-local-plugins"
python scripts/list_external_plugins.py --plugin-root $pluginRoot --validate

python scripts/run_external_strategy.py `
  --plugin-root $pluginRoot `
  --strategy qlora `
  --check-only
```

插件策略包括 `qlora`、`dora`、`adalora`、`ia3`、`partial_unfreeze`、`full_sft` 和
`prompt_tuning`。先对每个策略执行 `--check-only`，它会验证 manifest、平台、CUDA 和
独立依赖；缺少依赖时不会静默回退到 LoRA。

```powershell
python scripts/run_external_strategy.py `
  --plugin-root $pluginRoot `
  --strategy qlora `
  --config "$pluginRoot\\plugins\\qlora\\configs\\smoke.yaml" `
  --model-dir $env:LOCAL_MODEL_DIR `
  --processor-dir $env:LOCAL_MODEL_DIR `
  --train-file $env:TRAIN_JSONL `
  --val-file $env:VAL_JSONL `
  --image-root $env:DATA_ROOT `
  --max-steps 2 `
  --output-dir checkpoints/plugins/qlora/local-real-smoke
```

将 `qlora` 和配置路径替换为目标策略即可。插件目录是显式加载的本地实验包，默认不安装
缺失依赖，也不联网；只有确认版本兼容且准备好离线 wheel 后才使用 `--install-missing`
与 `--offline --wheel-dir <dir>`。Full SFT 和 Prompt Tuning 的资源/兼容性门槛最高，先
使用各自的 dry-run 或 forward-only 配置。

## 15. AutoDL 环境初始化

```bash
bash scripts/environment/setup_autodl.sh \
  --env-name rs-vlm --clone-current --install-dev --install-model
```

克隆当前镜像环境以保留 PyTorch/CUDA，安装项目和非 Torch 模型依赖，配置缓存目录。

## 16. 加载 AutoDL 环境变量

```bash
source /root/autodl_env.sh
```

设置项目、数据、模型、输出、临时目录及 Hugging Face、Torch、pip 缓存位置。

## 17. 同步数据到高速本地盘

```bash
bash scripts/storage/sync_to_local_disk.sh \
  --source /root/autodl-fs/datasets/vrsbench \
  --destination /root/autodl-tmp/packages/vrsbench
```

使用 rsync 断点续传和校验，避免训练时从网络盘高频读取大量小图片。

## 18. 解压数据

```bash
python scripts/data/unpack_dataset.py \
  --archive /root/autodl-tmp/packages/vrsbench/vrsbench_raw_v1.tar.zst \
  --destination /root/autodl-tmp/datasets
```

先验证旁路 SHA-256，再拒绝不安全归档成员并恢复原目录层级。

## 19. 云端环境检查

```bash
conda activate rs-vlm
python scripts/environment/check_environment.py --require-model --require-gpu
```

正式训练前确认模型栈、CUDA、BF16 能力和路径，而不是依赖镜像名称推测。

## 20. 云端真实 smoke

```bash
python scripts/training/run_train.py \
  --env-config configs/cloud/autodl.yaml \
  --config configs/cloud/train_lora_autodl_smoke.yaml \
  --environment autodl
```

使用 16 个左右样本和少量 step 检查真实前后向、保存与恢复。

## 21. 云端正式训练

```bash
bash scripts/training/run_autodl_train.sh \
  --config configs/cloud/train_lora_autodl.yaml
```

脚本依次检查环境和数据、运行 smoke、启动正式训练，并把输出写入带时间戳日志。

## 22. 使用 screen

```bash
screen -S qwen_lora
```

按 `Ctrl+A` 后按 `D` 退出但保持任务，`screen -r qwen_lora` 恢复，
`screen -ls` 查看会话。

## 23. 查看训练日志

```bash
tail -f /root/autodl-tmp/outputs/logs/train.log
```

实际脚本日志带时间戳；可先用 `ls -t "$OUTPUT_ROOT/logs" | head` 找最新文件。

## 24. 监控 GPU

```bash
watch -n 1 nvidia-smi
```

观察显存、利用率和进程；它只是监控命令，不替代训练报告中的峰值统计。

## 25. 断点续训

```bash
bash scripts/training/run_autodl_train.sh \
  --config configs/cloud/train_lora_autodl.yaml \
  --skip-smoke \
  --resume /root/autodl-tmp/outputs/<experiment>/checkpoints/checkpoint-1000
```

恢复 Trainer 的 optimizer、scheduler、global step 和随机状态，并继续写入原实验。

## 26. 备份结果

```bash
bash scripts/storage/backup_results.sh \
  --experiment-dir /root/autodl-tmp/outputs/<experiment> \
  --backup-root /root/autodl-fs/experiments
```

只备份配置、环境、日志、指标、预测、处理器、adapter 和最新若干 checkpoint。

## 27. 成功后自动关机

```bash
bash scripts/training/run_autodl_train.sh \
  --config configs/cloud/train_lora_autodl.yaml \
  --backup-after-train --auto-shutdown
```

只有 smoke、正式训练和备份全部成功才调用关机；`--auto-shutdown` 不允许脱离备份单独使用。

## 28. AutoDL RTX 4090 LoRA 参数配置

`dev-dqt` 的完整 4090 参数已整理为无绝对路径配置，继续使用现有稳定 LoRA 脚本：

```bash
source environments/autodl.env
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_autodl_4090.yaml
```

配置默认 batch size 为 16、bf16、gradient checkpointing、cosine scheduler，并把输出写入
`${OUTPUT_ROOT}/checkpoints/lora/autodl_4090_full`。运行前应先执行环境和资产检查；显存不足时
优先降低 `per_device_train_batch_size` 并提高 `gradient_accumulation_steps`，保持有效 batch
size，而不是修改已验证的数据格式或 LoRA target modules。
