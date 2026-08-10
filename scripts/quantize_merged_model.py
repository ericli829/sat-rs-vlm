"""兼容 dev-dqt 命令：量化配置中 ``model.merged_model`` 指向 merge 后模型。"""

from __future__ import annotations

import warnings

if __package__:
    from scripts.quantize_rs_vlm import main
else:
    from quantize_rs_vlm import main


if __name__ == "__main__":
    warnings.warn(
        "quantize_merged_model.py is a compatibility wrapper; use quantize_rs_vlm.py.",
        DeprecationWarning,
        stacklevel=1,
    )
    raise SystemExit(main(forced_backend="torch_dynamic_int8"))
