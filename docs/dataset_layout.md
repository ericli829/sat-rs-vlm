# 数据集布局

## VRSBench

保留下载后的原始结构，只增加：

```text
VRSBench/
├── <原始目录，保持不变>
└── project_metadata/
    ├── dataset_manifest.json
    ├── train.jsonl
    ├── validation.jsonl
    ├── test.jsonl
    ├── smoke.jsonl
    ├── statistics.json
    └── splits/
```

Manifest 声明数据版本、坐标格式、坐标范围和分片相对路径。JSONL 的 `images`
同样为相对路径，加载器用 `dataset_root / relative_path` 解析。

## 校验

校验器检查 manifest、四个分片、绝对 Windows 路径、重复 ID、分片交叉、图片存在
与可解码性、任务字段和 xyxy bbox 范围，并输出样本数、图片数和任务分布。

```bash
python scripts/data/validate_dataset.py \
  --dataset-root /path/to/VRSBench \
  --manifest-name project_metadata/dataset_manifest.json
```

## 搬运

```bash
python scripts/data/package_dataset.py \
  --dataset-root /path/to/VRSBench \
  --output /path/to/vrsbench_v1.tar.gz

python scripts/data/unpack_dataset.py \
  --archive /path/to/vrsbench_v1.tar.gz \
  --destination /path/to/datasets
```

`.tar.zst` 需要系统 `zstd`。两种格式都生成并校验 SHA-256；解压拒绝路径穿越和链接。
