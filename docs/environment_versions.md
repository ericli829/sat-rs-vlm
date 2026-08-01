# 环境版本策略

详细机器可读记录见 `environments/versions.yaml`。

已成功跑通 LoRA 的参考栈是 Python 3.11、Torch 2.12.1、Transformers 5.13.0、
PEFT 0.19.1、Accelerate 1.14.0 和 bitsandbytes 0.49.2。当前 CPU 默认验证环境是
Anaconda Python 3.12.4；模型依赖为可选，不作为默认单元测试前提。

原则：

1. `pyproject.toml` 是依赖真源。
2. AutoDL 镜像的 CUDA 匹配 Torch 优先，不由 setup 脚本覆盖。
3. 版本能力在运行前探测，不因可选策略强制升级整个环境。
4. 每次正式训练保存 `environment.json`，需要完整冻结时另运行
   `scripts/environment/export_environment.py`。
