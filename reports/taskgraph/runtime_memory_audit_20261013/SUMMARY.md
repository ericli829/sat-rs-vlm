# 生产路由记忆审计

审计了原表 `configs/taskgraph/taskgraph_qwen_test.jsonl` 的 3,738 条记录。原文件字节级备份为 `original_memory.jsonl`，SHA256 为 `f649cf1b972dc06e339d8c17c69733f7ef43017ce8f6192399b2f0e23dee2cbe`；原配置表未覆盖。逐条判断索引见 `audit_index.jsonl`，机器统计见 `summary.json`。

表中 2,410 条标记 `DIRECT_VLM`，1,328 条标记 `TASKGRAPH_UHR`；全部 3,738 条均缺少模型角色、正确性状态和独立评价证据。重复/冲突图像与问题哈希组有 1 组、涉及 2 条记录。由于 `DIRECT_VLM` 本身不能证明 2B 回答正确，最终仅保留 **0 条**，新生产白名单 `configs/taskgraph/taskgraph_qwen_verified.jsonl` 为空。生产配置已改为只读该白名单、要求已验证记录并关闭自动写入。

新白名单仍使用按顺序的图片 SHA256 与原始问题 SHA256 精确匹配。严格模式只接受显式标记为 `semantic_2b`、`verified_correct` 且附带可追溯评价文件 SHA256 与同一 sample_id 的直接 VLM 记录；重复 ID 或重复/冲突哈希不会命中。
