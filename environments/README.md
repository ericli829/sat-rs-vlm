# 环境与依赖

`pyproject.toml` 是依赖真源。本目录提供安装工具兼容视图和已验证版本记录：

- `versions.yaml`：已跑通 LoRA 环境、当前默认验证环境和最低版本。
- `requirements-base.txt`：CLI、HTTP、配置和数据工具。
- `requirements-dev.txt`：pytest、ruff、mypy。
- `requirements-model.txt`：不含 PyTorch 的模型栈，保护云镜像 CUDA 兼容性。
- `requirements-cloud.txt`：TensorBoard 等云端可选能力。
- `requirements-qlora.txt`：仅 QLoRA 和 `bnb_int8` 需要的 bitsandbytes。
- `constraints.txt`：已跑通 LoRA 环境的参考版本，不应盲目覆盖云端 Torch。
- `*.env.example`：本地和 AutoDL 路径变量模板，不包含密钥。

QLoRA 的 bitsandbytes 是 `pyproject.toml` 中独立的 `qlora` 可选依赖；AutoDL 可通过
`setup_autodl.sh --install-qlora` 安装，不影响 LoRA 和基础测试。
