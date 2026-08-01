# 故障排查

## `ModuleNotFoundError: sat_rs_vlm`

从项目根目录执行并安装 editable 包：

```bash
python -m pip install -e .
python -c "import sat_rs_vlm; print(sat_rs_vlm.__file__)"
```

脚本入口也会把项目 `src` 加入路径，不能用错误解释器调用已安装在其他环境的包。

## 基础依赖缺失

```bash
python scripts/environment/check_environment.py
python -m pip install -e ".[dev]"
```

确认 `python -c "import sys; print(sys.executable)"` 与预期环境一致。

## 数据校验失败

不要移动原始 VRSBench 文件来迁就配置。修正 manifest/JSONL 的相对路径，并检查
`DATA_ROOT`。绝对盘符、bbox 越界、重复 ID 和 split 交叉都会明确列出。

## CUDA 设备不一致或输出为空

先运行 `--forward-only`；检查模型和 batch 是否被移动到同一设备。项目稳定训练与
评估脚本已有 `model_input_device` 和 `move_to_device` 逻辑。不要把 CPU tensor
直接传给 CUDA 模型。

## BF16 不支持

统一入口在自动模式下使用 CUDA BF16（支持时）或 FP16；CPU 使用 FP32。显式同时
开启 BF16/FP16 会失败。云端执行环境报告确认 `bf16_supported`。

## 断点找不到

`--resume-from-checkpoint` 必须指向真实目录；`--resume-latest` 只搜索当前实验的
`checkpoints/checkpoint-*`。不要把 adapter 根目录误当 Trainer checkpoint。

## AutoDL 系统盘变满

确认 HF、Torch、pip 和 TMPDIR 都由 `/root/autodl_env.sh` 指向
`/root/autodl-tmp`。训练后使用选择性备份脚本，不复制全部中间 checkpoint。
