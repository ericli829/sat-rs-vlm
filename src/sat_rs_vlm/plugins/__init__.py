"""可选外部微调插件 API；导入时不执行发现或加载。"""

from sat_rs_vlm.plugins.api import EXTERNAL_PLUGIN_API_VERSION, ExternalFineTuningPlugin
from sat_rs_vlm.plugins.cli import run_local_plugin_command
from sat_rs_vlm.plugins.context import PluginContext
from sat_rs_vlm.plugins.errors import (
    PluginCompatibilityError,
    PluginDependencyError,
    PluginExecutionError,
    PluginValidationError,
)

__all__ = [
    "EXTERNAL_PLUGIN_API_VERSION",
    "ExternalFineTuningPlugin",
    "PluginCompatibilityError",
    "PluginContext",
    "PluginDependencyError",
    "PluginExecutionError",
    "PluginValidationError",
    "run_local_plugin_command",
]
