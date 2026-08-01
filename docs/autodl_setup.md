# AutoDL 环境

## 存储约定

- `/root/autodl-tmp`：训练时的高速数据、模型、缓存、输出和临时文件。
- `/root/autodl-fs`：数据归档、重要 checkpoint 和实验备份。
- `/root/tf-logs`：TensorBoard 日志。

这些路径仅位于云配置、环境模板和 shell 脚本，不散落在 Python 业务代码。

## 初始化

```bash
bash scripts/environment/setup_autodl.sh \
  --env-name rs-vlm --clone-current --install-dev --install-model
source /root/autodl_env.sh
conda activate rs-vlm
```

`--clone-current` 优先保留镜像当前 PyTorch/CUDA 组合。脚本以 `--no-deps` 安装项目，
再安装不包含 Torch 的模型 requirements；不会盲目升级 PyTorch。
脚本会安装 `requirements-cloud.txt`。若要运行 QLoRA 或 CUDA `bnb_int8`，使用：

```bash
bash scripts/environment/setup_autodl.sh \
  --env-name rs-vlm --clone-current --install-dev --install-model --install-qlora
python scripts/environment/check_environment.py \
  --require-model --require-bitsandbytes --require-gpu
```

`--install-qlora` 只增加独立的 `requirements-qlora.txt`，不会让基础 LoRA 和 CPU 测试强制依赖
bitsandbytes。

`--require-model` 会同时检查 Torch、与镜像 Torch 匹配的 torchvision、Transformers、PEFT、
Accelerate、safetensors 和 qwen-vl-utils。脚本不会擅自替换 CUDA 镜像中的 Torch/torchvision；
若预检报告二者缺失或版本不匹配，应安装云镜像 CUDA 版本对应的官方组合后再继续。

## 训练前

```bash
python scripts/environment/check_environment.py --require-model --require-gpu
python scripts/data/validate_dataset.py \
  --dataset-root "$DATA_ROOT/VRSBench" \
  --manifest-name project_metadata/dataset_manifest.json
```

正式脚本默认先执行真实模型 smoke，失败时不会进入正式训练：

```bash
bash scripts/training/run_autodl_train.sh \
  --config configs/cloud/train_lora_autodl.yaml
```

最终 Adapter 位于运行目录的 `checkpoints/` 根层，并包含 `strategy_manifest.json`、
`adapter_config.json`、Adapter 权重和 `processor/`。统一评估和可靠性命令应指向这一根层，
而不是其中的 Trainer `checkpoint-*` 子目录。

`--auto-shutdown` 必须与 `--backup-after-train` 同时使用，且只在训练及备份均成功后
调用关机。首次上云必须人工验证 shell、CUDA、模型路径和恢复流程。
