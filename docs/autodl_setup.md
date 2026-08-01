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

`--auto-shutdown` 必须与 `--backup-after-train` 同时使用，且只在训练及备份均成功后
调用关机。首次上云必须人工验证 shell、CUDA、模型路径和恢复流程。
