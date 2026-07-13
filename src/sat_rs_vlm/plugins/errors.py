"""外部插件公开异常类型。"""

from __future__ import annotations


class ExternalPluginError(RuntimeError):
    """包含插件、阶段、原因和建议动作的统一异常。"""

    def __init__(
        self,
        *,
        plugin_name: str,
        stage: str,
        reason: str,
        suggested_action: str,
    ) -> None:
        self.plugin_name = plugin_name
        self.stage = stage
        self.reason = reason
        self.suggested_action = suggested_action
        super().__init__(
            f"plugin={plugin_name}; stage={stage}; reason={reason}; "
            f"suggested_action={suggested_action}"
        )


class PluginValidationError(ExternalPluginError):
    """manifest、配置或路径不合法。"""


class PluginDependencyError(ExternalPluginError):
    """插件依赖缺失、冲突或安装失败。"""


class PluginCompatibilityError(ExternalPluginError):
    """API、平台、CUDA 或 Python 版本不兼容。"""


class PluginExecutionError(ExternalPluginError):
    """入口实例化或插件运行失败。"""
