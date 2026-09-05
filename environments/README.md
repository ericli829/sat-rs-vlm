# 环境与依赖

`pyproject.toml` 是依赖真源。本目录提供安装工具兼容视图和已验证版本记录：

- `versions.yaml`：已跑通 LoRA 环境、当前默认验证环境和最低版本。
- `requirements-base.txt`：CLI、HTTP、配置和数据工具。
- `requirements-dev.txt`：pytest、ruff、mypy。
- `requirements-model.txt`：不含 PyTorch 的模型栈，保护云镜像 CUDA 兼容性。
- `requirements-cloud.txt`：TensorBoard 等云端可选能力。
- `requirements-qlora.txt`：仅 QLoRA 和 `bnb_int8` 需要的 bitsandbytes。
- `pyproject.toml[retriever]`：GeoRSCLIP/OpenCLIP 的 `open_clip_torch` 与 `timm`。
- `constraints.txt`：已跑通 LoRA 环境的参考版本，不应盲目覆盖云端 Torch。
- `*.env.example`：本地和 AutoDL 路径变量模板，不包含密钥。

QLoRA 的 bitsandbytes 是 `pyproject.toml` 中独立的 `qlora` 可选依赖；AutoDL 可通过
`setup_autodl.sh --install-qlora` 安装，不影响 LoRA 和基础测试。

本地统一环境可按需安装 Qwen 与 GeoRSCLIP：

```bash
python scripts/environment/bootstrap_local.py \
  --with-dev \
  --with-model \
  --with-retriever
python scripts/environment/check_environment.py \
  --require-model \
  --require-retriever
```

AutoDL 的完整 TaskGraph bootstrap：

```bash
bash scripts/environment/setup_autodl.sh \
  --install-model \
  --install-retriever \
  --install-lae
```

`--install-retriever` 直接复用 `pyproject.toml` 的 `retriever` extra，且把当前
Torch 完整版本写入临时 pip constraint；脚本不使用 `--upgrade` 或
`--force-reinstall`，因此不会替换云镜像已有的 CUDA Torch。运行前必须已有可 import
的 Torch。首次创建主环境并选择 model/retriever 时，脚本会克隆当前含 CUDA Torch
的 Conda 环境；不会在空环境里从 PyPI 猜测 Torch wheel。`--dry-run` 可在不创建
环境、不安装包的情况下查看完整计划。

LAE-DINO 始终安装到独立 Conda 环境（默认 `rs-vlm-lae`），不会把 MMCV、
MMEngine 或 MMDetection 安装进主 `rs-vlm`。安装所需源码、config、checkpoint 与
BERT 都必须预先存在；bootstrap 不下载模型或权重。路径可通过 `LAE_DINO_*`
环境变量或 `setup_autodl.sh --lae-*` 参数提供。成功后 sidecar 所需变量会写入
`/root/autodl_env.sh`。
