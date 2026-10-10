# 推送前独立快照验证

2026-10-10。仅暂存LOCATE/SELECT重构；runtime.py按修改块分离，原hash-memory/crop-policy工作继续留在未暂存工作树。未暂存competition API、HF engine/cache、memory表与相关其他修改。

使用 `git write-tree` + `git archive` 导出待提交索引的src、必要Lab/提示词/审计脚本及新增测试，验证不依赖工作树中未提交的HF engine和memory实现。

```powershell
& 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe' -m pytest -q --basetemp=D:/Desktop/tzb-2026/sat-rs-vlm/.tmp/v2-push-index-check tests/unit/taskgraph/test_locate_routing_v2.py tests/unit/taskgraph/test_locate_select_v2.py
```

结果：**56 passed in 7.43s**。独立快照的runtime.py Ruff通过。此前工作树分批回归617项通过，详情见manifest.json；56项为重复验证，不追加到617的唯一测试总数。

首次快照测试因basetemp父目录不存在出现fixture setup错误，改用已有主仓库.tmp下的唯一basetemp后全部通过，没有代码失败。暂存区空白检查排除了两份逐字复制的原始Qwen chat_template.json；它们原有文件末尾空格/空行保留作审计，未改写原始模板。其余暂存文件检查通过。

完整日志见push_verification.log。导出目录只用于验证，不是新Git工作树。
