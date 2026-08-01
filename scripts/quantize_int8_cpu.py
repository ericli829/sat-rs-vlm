"""已弃用：兼容旧 CPU INT8 命令，请改用 quantize_rs_vlm.py。"""

from __future__ import annotations

import warnings

if __package__:
    from scripts.quantize_rs_vlm import main
else:
    from quantize_rs_vlm import main

if __name__ == "__main__":
    warnings.warn(
        "quantize_int8_cpu.py is deprecated; use quantize_rs_vlm.py --backend torch_dynamic_int8",
        DeprecationWarning,
        stacklevel=1,
    )
    raise SystemExit(main(forced_backend="torch_dynamic_int8"))
