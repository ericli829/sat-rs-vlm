# 实验工作流

1. 设置 `DATA_ROOT`、`MODEL_ROOT` 和 `OUTPUT_ROOT`。
2. 运行环境检查与数据 manifest 校验。
3. 本地运行 Mock smoke，验证配置合并、分片读取和输出目录。
4. 云端运行真实模型 smoke，验证前向、反向、保存和恢复。
5. 使用 `configs/experiments/` 中的实验配置启动正式训练。
6. 从原实验目录显式或以 `latest` 恢复，避免创建无关新实验。
7. 使用统一评估入口保存 predictions 和 metrics。
8. 备份配置、环境、日志、指标、adapter 和最新若干 checkpoint。

每次实验必须保留 resolved config、命令、Git commit、环境快照和 preflight 报告。
不同数据版本、seed 或训练步数的指标不可直接当作同一对照实验。
