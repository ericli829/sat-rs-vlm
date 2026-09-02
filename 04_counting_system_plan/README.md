# Counting System（TaskGraph COUNT）

实现计划书 `04_counting_system_plan.md`：在高分辨率整图或指定 Region 上做可靠计数。

```text
scope → ScalePolicy → tiling → LAE-DINO → local→global bbox
      → same-scale Core Ownership + NMS → cross-scale fusion → count
```

EntitySet 输入时 **不重新检测**，`COUNT = len(EntitySet)`。

## 资源路径

Python 只读 `configs/paths.yaml` 和环境变量，不写死 AutoDL 路径。
`configs/paths.yaml` 里仍写 `/root/autodl-fs`；`counting_system.paths` 会映射到本机：

- AutoDL：目录存在则原样使用 `/root/autodl-fs`、`/root/autodl-tmp`
- Windows / 本机：仓库上一级 `autodl-fs/`、`autodl-tmp/`（本仓库即 `E:\26揭榜挂帅-太空智算\autodl-fs`）
- 也可设 `COUNTING_AUTODL_FS` / `COUNTING_AUTODL_TMP`

| 用途 | 逻辑路径（yaml） | 实际落点 |
|---|---|---|
| XLRS-Bench-lite 计数子集 | `/root/autodl-fs/datasets/xlrsbench-lite` | autodl-fs 或 autodl-tmp 符号链接 |
| LAE-DINO 仓库 + 权重 | `/root/autodl-fs/models/lae-dino` | 同上 |
| GeoRSCLIP ViT-B-32 | `/root/autodl-fs/models/georsclip/ckpt/RS5M_ViT-B-32.pt` | 同上 |

完整 `initiacms/XLRS-Bench-lite` 约 43GB / 3080 题。下载脚本流式扫描后只导出 **Counting** 样本，当前落盘 **320 条**（Overall 60 / Regional 100 / complex reasoning 100 / change detection 60，变化检测双图都会保存，图像约 5.9GB）。CLIP 必须用 **GeoRSCLIP**（`RS5M_ViT-B-32.pt`），不是 RemoteCLIP。

环境变量可覆盖：`COUNTING_DATASET_ROOT`、`COUNTING_LAE_WEIGHTS`、`COUNTING_GEORS_CKPT`、`HF_ENDPOINT`。国内默认 `https://hf-mirror.com`。

## 安装与下载

```bash
cd 04_counting_system_plan
pip install -r requirements.txt
# --xlrs-max 0 表示导出全部计数样本（不要再用 24 条冒充全集）
python scripts/download_assets.py --xlrs-max 0
```

`download_assets.py` 会：

1. clone [LAE-DINO](https://github.com/jaychempan/LAE-DINO)（git 失败则下 zip）
2. 下载 `lae_dino_swint_lae1m-28ca3a15.pth` 与 `bert-base-uncased`
3. 下载 GeoRSCLIP `RS5M_ViT-B-32.pt`
4. 流式导出 XLRS-Bench-lite 全部计数样本到 autodl-fs

## 运行

```bash
# 1. Fake E2E（无 GPU / 无权重）
python scripts/run_fake_e2e.py

# 2. 单图计数（真实检测器：优先 LAE-DINO mmdet，否则 GroundingDINO）
python scripts/run_count.py --image /path/to/uhr.jpg --target ship --entire --backend auto

# 3. Region 穷尽计数
python scripts/run_count.py --image /path/to/uhr.jpg --target airplane --region TOP --no-entire

# 4. XLRS 计数 benchmark
python scripts/run_benchmark.py --max-samples 8 --backend auto

# 5. 计划书第 13 节消融（scale / threshold / prompt / tile / gate）
python scripts/run_experiments.py

# 单元测试
pytest -q
```

可选 GeoRSCLIP coarse gate（`entire=true`，recall-first，不是 Top-K）：

```bash
python scripts/run_count.py --image uhr.jpg --target ship --entire --gate
```

## 接口

```python
from counting_system import CountExecutor, ImageRef, EntitySet

result = CountExecutor()(ImageRef(path="a.jpg", width=W, height=H), "ship", entire=True)
result.to_scalar()  # ScalarInt
# result.detections / result.provenance 进入 trace，不进 TaskGraph 对外接口
```

- `entire=true`：整图，可开 CLIP gate，再 native / 可选 fine。
- `entire=false`：当前 Region 内 exhaustive，不做 Top-K。
- Fine 是否开启由类别 tiny profile 决定，不用“当前置信度低”当唯一依据。
- 原始 proposals 一律写入 provenance，threshold sweep 区分“没生成”和“被后处理截掉”。

## 检测后端

| backend | 说明 |
|---|---|
| `lae_mmdet` | 官方 LAE-DINO（需要 mmdet/mmcv + 预训练权重） |
| `grounding_dino` | Transformers GroundingDINO，同族开放词汇检测，当前 CUDA 13 镜像可跑 |
| `fake` | 连通域合成检测，用于 Fake E2E |
| `auto` | 有 mmdet 则 LAE，否则 GroundingDINO |

官方 LAE 权重已按计划下载；在 RTX 5090 / torch 2.12 上 mmcv 旧栈通常装不上，所以 `auto` 会回退到 GroundingDINO，provenance 会标明 backend。

## 验收对照

| 计划书 | 状态 |
|---|---|
| Fake E2E | `scripts/run_fake_e2e.py` + `tests/test_fake_e2e.py` |
| 真实检测可跑 | `LAEDinoDetector` |
| Region count | `--region` / `entire=false` |
| UHR tiled count | native tile + ownership core |
| global bbox 映射 | `local_to_global` |
| 去重稳定 | Core Ownership + NMS + 跨尺度融合 |
| overlay + JSON trace | `overlay.py` / `trace.py` |
| XLRS 真实计数样本 | `scripts/run_benchmark.py` |
