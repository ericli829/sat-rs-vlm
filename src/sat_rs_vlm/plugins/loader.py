"""使用标准库从普通文件夹安全加载插件入口。"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import traceback
from types import ModuleType

from sat_rs_vlm.plugins.api import EXTERNAL_PLUGIN_API_VERSION, ExternalFineTuningPlugin
from sat_rs_vlm.plugins.discovery import DiscoveredPlugin
from sat_rs_vlm.plugins.errors import PluginCompatibilityError, PluginExecutionError
from sat_rs_vlm.plugins.manifest import resolve_inside


def _unique_module_name(plugin: DiscoveredPlugin) -> str:
    digest = hashlib.sha256(str(plugin.directory).encode("utf-8")).hexdigest()[:12]
    safe_name = plugin.manifest.plugin.name.replace("-", "_")
    return f"sat_rs_vlm_external_plugin_{safe_name}_{digest}"


def _load_module(plugin: DiscoveredPlugin) -> ModuleType:
    entrypoint = resolve_inside(
        plugin.directory,
        plugin.manifest.entrypoint.module,
        label="entrypoint.module",
    )
    if not entrypoint.is_file():
        raise PluginExecutionError(
            plugin_name=plugin.manifest.plugin.name,
            stage="load_entrypoint",
            reason=f"entrypoint file does not exist: {entrypoint}",
            suggested_action="Fix entrypoint.module in plugin.yaml.",
        )
    module_name = _unique_module_name(plugin)
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if spec is None or spec.loader is None:
        raise PluginExecutionError(
            plugin_name=plugin.manifest.plugin.name,
            stage="load_entrypoint",
            reason=f"cannot create import spec for {entrypoint}",
            suggested_action="Use a normal Python source file as entrypoint.",
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        summary = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        raise PluginExecutionError(
            plugin_name=plugin.manifest.plugin.name,
            stage="load_entrypoint",
            reason=summary,
            suggested_action="Inspect the original chained exception and plugin entrypoint.",
        ) from exc
    # Keep the unique module available for traceback/source inspection without changing sys.path.
    sys.modules[module_name] = module
    return module


def load_external_plugin(plugin: DiscoveredPlugin) -> ExternalFineTuningPlugin:
    """加载入口类并验证 API、name 和 version 与 manifest 一致。"""

    module = _load_module(plugin)
    class_name = plugin.manifest.entrypoint.class_name
    plugin_class = getattr(module, class_name, None)
    if not isinstance(plugin_class, type) or not issubclass(plugin_class, ExternalFineTuningPlugin):
        raise PluginCompatibilityError(
            plugin_name=plugin.manifest.plugin.name,
            stage="load_entrypoint",
            reason=f"{class_name!r} is not an ExternalFineTuningPlugin subclass",
            suggested_action="Subclass sat_rs_vlm.plugins.ExternalFineTuningPlugin.",
        )
    try:
        instance = plugin_class()
    except Exception as exc:
        summary = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        raise PluginExecutionError(
            plugin_name=plugin.manifest.plugin.name,
            stage="instantiate",
            reason=summary,
            suggested_action="Make the plugin constructor argument-free and inspect the cause.",
        ) from exc
    expected = plugin.manifest.plugin
    actual = {
        "name": getattr(instance, "name", None),
        "version": getattr(instance, "version", None),
        "api_version": getattr(instance, "api_version", None),
    }
    required = {
        "name": expected.name,
        "version": expected.version,
        "api_version": EXTERNAL_PLUGIN_API_VERSION,
    }
    mismatches = [key for key, value in required.items() if actual[key] != value]
    if mismatches:
        raise PluginCompatibilityError(
            plugin_name=expected.name,
            stage="api_validation",
            reason=f"entrypoint/manifest mismatch for {mismatches}: {actual}",
            suggested_action="Align class attributes with plugin.yaml.",
        )
    return instance
