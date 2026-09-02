# Counting System（TaskGraph COUNT）

实现计划书 `04_counting_system_plan.md`：在高分辨率整图或指定 Region 上做可靠计数。

本目录是 counting **算法 backend / 研究工作区**。TaskGraph 对外合同以
`src/sat_rs_vlm/taskgraph` 为准，见 [TASKGRAPH.md](TASKGRAPH.md)。

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

# 4. XLRS 计数 benchmark（默认 tile 1333/0.15 对齐 uhr-locator）
python scripts/run_benchmark.py --max-samples 8 --backend auto

# 5. 转为 sat-rs-vlm Evaluation v1.5 predictions（可离线 evaluate_predictions.py）
python scripts/export_predictions_v15.py --input outputs/xlrs_benchmark/predictions.jsonl

# 全量 320 条 + 完整 protocol 报告（初赛材料）
python scripts/run_benchmark.py --max-samples 0 --backend auto --no-overlay --out outputs/xlrs_benchmark_all

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

## 与 feature/vlm-semantic-alignment 对齐

上游分支：[sat-rs-vlm/feature/vlm-semantic-alignment](https://github.com/ericli829/sat-rs-vlm/tree/feature/vlm-semantic-alignment)

| 上游接口 | 本仓库对应 |
|---|---|
| `COUNT` exactly_one `image` / `entities` | `contracts.validate_count_inputs` + `CountExecutor.execute` |
| `CountParams(target, entire)` | `executor.CountParams` |
| `TargetSpec.category` + `phrase()` | `target.TargetSpec` |
| `SelectResult` EMPTY → 0 | `unwrap_select_result(allow_empty=True)` |
| 输出 `ScalarInt` | `CountExecutor.execute` / `CountResult.to_scalar` |
| tiled 1333 / overlap 0.15 / NMS 0.4 | `configs/default.yaml` native tile |
| 全局坐标 dedup，禁止 per-crop 相加 | `fusion.py` Core Ownership + NMS |
| `counting_protocol.py` Exact Match / RMSE | `predictions_v15.jsonl` |

离线接入主评测：

```bash
# 在 sat-rs-vlm（feature/uhr-locator）仓库内
python scripts/evaluation/evaluate_predictions.py \
  --config configs/eval/evaluation_v1_5.yaml \
  --predictions /path/to/04_counting_system_plan/outputs/xlrs_benchmark/predictions_v15.jsonl \
  --output-dir reports/research/xlrs_counting_detector
```

## 初赛答疑对齐（20260801）

| 要求 | 本仓库做法 |
|---|---|
| 主指标 Exact Match / RMSE | `eval/metrics.py` → `exact_match`、`rmse` |
| 不用 GPT Judge 作主结果 | 检测计数路径，无 LLM 裁判 |
| 披露 tile / 阈值 / prompt | `protocol.json` + `configs/default.yaml` |
| 32B 参数量上限 | `eval/protocol.py` → `resources.within_32b_limit` |
| 峰值显存 / e2e 时间 / 冷启动 | `metrics.json` + `protocol.json` → `timing`、`gpu` |
| 分任务类型分项报告 | `protocol.json` → `by_category` |
| lite 子集 vs 官方全集分开标注 | `protocol.official_aligned=false`（默认） |

Benchmark 输出：

```text
outputs/xlrs_benchmark/
  metrics.json           # 汇总指标 + GPU
  protocol.json          # 协议披露、分任务、资源、环境
  predictions.jsonl      # 逐样本 pred/ref（模块内部格式）
  predictions_v15.jsonl  # sat-rs-vlm Evaluation v1.5 兼容格式
  trace.jsonl            # 检测 provenance
```
