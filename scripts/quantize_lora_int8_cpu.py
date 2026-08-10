"""兼容 dev-dqt 命令：对已 merge 的 LoRA 模型运行 CPU dynamic INT8。"""

from __future__ import annotations

import warnings

if __package__:
    from scripts.quantize_rs_vlm import main
else:
    from quantize_rs_vlm import main


if __name__ == "__main__":
    warnings.warn(
        "Merge the LoRA adapter first, then use quantize_rs_vlm.py; "
        "unmerged LoRA + torch dynamic INT8 is intentionally rejected.",
        DeprecationWarning,
        stacklevel=1,
    )
    raise SystemExit(main(forced_backend="torch_dynamic_int8"))
