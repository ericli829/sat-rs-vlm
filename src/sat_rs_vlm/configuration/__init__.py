"""本地与云端共用的轻量级配置系统。"""

from sat_rs_vlm.configuration.layered import (
    LayeredConfigRequest,
    load_layered_config,
    write_resolved_config,
)
from sat_rs_vlm.configuration.merge import deep_merge, set_dotted_value
from sat_rs_vlm.configuration.paths import PathConfig, resolve_path_value
from sat_rs_vlm.configuration.precision import PrecisionDecision, select_precision

__all__ = [
    "LayeredConfigRequest",
    "PathConfig",
    "PrecisionDecision",
    "deep_merge",
    "load_layered_config",
    "resolve_path_value",
    "select_precision",
    "set_dotted_value",
    "write_resolved_config",
]
