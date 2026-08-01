# 数据目录

仓库只保存 manifest schema、示例 manifest 和微型测试夹具，不提交完整遥感数据。

## 原始数据原则

VRSBench 的原始文件名、目录层级、图片与标注对应关系都不修改。项目生成的
JSONL、划分、统计和 manifest 统一放在 `VRSBench/project_metadata/`。样本图片
字段必须是相对数据集根目录的 POSIX 风格路径，禁止保存本机盘符或 `/root` 路径。

## Manifest

`data/manifests/dataset_manifest.schema.json` 是版本 1 schema；
`data/manifests/example_manifest.json` 是 VRSBench 示例。`coordinate_range` 可声明
`[0, 1]` 或其他明确范围，校验器按 manifest 执行，不隐式猜测。

## 常用命令

```bash
python scripts/data/prepare_dataset.py --help
python scripts/data/validate_dataset.py --dataset-root tests/fixtures/miniature_dataset
python scripts/data/package_dataset.py --help
python scripts/data/unpack_dataset.py --help
```

归档和校验文件已被 Git 忽略。解压器拒绝路径穿越和符号链接。
