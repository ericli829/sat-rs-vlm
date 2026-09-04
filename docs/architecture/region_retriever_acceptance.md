# Region Retriever 工程验收与 GeoRSCLIP 定版

## 最终决定

Region Retriever 的生产默认模型为 **GeoRSCLIP ViT-B/32**。生产配置、TaskGraph
真实示例、缓存目录、环境变量、六图 UHR smoke 和本验收结论均已切换到
GeoRSCLIP。RemoteCLIP 仅作为可替换 provider 和历史对比模型保留，不再属于
默认运行路径。

模型选择来自队长的工程决定。离线指标用于描述行为与选择参数，不再用来推翻该
决定。尤其是 VRSBench 按 01 系统方针应走 Direct VLM/Detection；这里只把其
bbox 标注用作 Retriever 候选几何与排序的离线代理评测。

## 与 01/02 方针的一致性

- TaskGraph 只依赖 `RegionRetrieverProvider` capability，DAG 不出现 GeoRSCLIP、
  checkpoint、grid、阈值、设备等物理细节。
- Retriever 负责 UHR/LOCATE 的语义区域粗检索，不是 detector 或回答模型。
- 语义大区域由 LOCATE 路由到 Retriever；普通物体仍由 LAE-DINO 类 detector
  处理。
- Count 可复用 Retriever 作为 reject gate，但 gate 默认关闭，未达到部署域验收
  前不得启用。
- 输出支持多个候选，并包含 absolute global bbox、relevance score、provider、
  model、tile/level 和搜索 provenance。
- `ImageRef`、`Region`、`EntitySet` 等 typed runtime object 保持为模块边界。

## 02 交付验收

| # | 交付项 | 状态 | 实现/证据 |
|---:|---|---|---|
| 1 | RegionRetriever interface | 完成 | `taskgraph/providers.py` 的 request/candidate/provider contracts |
| 2 | Fake provider | 完成 | `FakeRegionRetriever`，离线测试无需真实权重 |
| 3 | 真实 baseline | 完成 | GeoRSCLIP OpenCLIP provider；真实权重加载键 0 missing / 0 unexpected |
| 4 | benchmark script | 完成 | fixed/sliding candidates、Recall@K、coverage、latency、peak VRAM、gate、CSV/JSON；GPU 数值由目标主机生成 |
| 5 | cache | 完成 | decoded image、image embedding、query embedding、atomic disk score cache、batch encode |
| 6 | overlay 可视化 | 完成 | 六张 XLRS 大图的 overview、ROI crops、comparison 和 contact sheet |
| 7 | 参数配置 | 完成 | mock 测试配置与 `uhr_hierarchical.georsclip.yaml` 生产配置分离 |
| 8 | 结果与推荐 | 完成 | GeoRSCLIP 定版、本文件、原始报告和两组真实六图结果 |

## 候选几何与离线结果

生产 UHR Locator 使用 3x3 core 和 `halo_ratio=0.25`。一级 core 是原图的 1/3，
加 halo 后观察窗约为原图的 1/2；随后只对高分区域继续分层 zoom。TaskGraph
直接 Retriever 示例使用等价的 3x3、`candidate_window_ratio=0.5`、最多 5 个候选。

GeoRSCLIP 在 corrected VRSBench-200 上的单窗 Recall：

| 候选协议 | 选择 | Coverage 50% | Coverage 80% | Coverage 90% |
|---|---:|---:|---:|---:|
| 3x3 不重叠 1/3 窗 | Top-5 | 64.0% | 38.5% | 32.5% |
| 3x3 重叠 1/2 窗 | Top-5 | 90.0% | 75.0% | 68.5% |
| 3x3 重叠 1/2 窗单窗 Oracle | 全部候选中取最佳 | 96.5% | 83.0% | 76.5% |
| 2x2 重叠 2/3 窗 | Top-3 | 98.0% | 89.5% | 84.5% |

2x2 ablation 的三个输出窗空间并集固定覆盖约 88.9% 原图，定位过粗，因此不作为
默认生产参数。3x3 重叠窗在候选粒度与高召回之间更平衡。

Benchmark 已区分以下面积口径：

- `mean_selected_roi_area_ratio`：单个选中 ROI 的平均面积比例；
- `selected_union_area_ratio`：选中 ROI 的真实几何并集，不重复计算重叠；
- `processed_area_ratio`：累计处理面积，允许大于 1；
- `topk_union_gt_coverage`：Top-K 与 GT 的真实几何并集覆盖。

旧 `selected_area_ratio` 暂时作为 `processed_area_ratio` 的兼容别名保留。此前将各
crop coverage 直接相加的伪 union 算法已修正。

## Cache、Count gate 与真实链路

- GeoRSCLIP 3x3 sliding-50、20 条/180 crop 的首轮平均延迟为 842.1 ms/样本；
  全部 score cache 命中后为 158.0 ms/样本，约 **5.33x**。召回和排序不变。
- GeoRSCLIP Count gate 在 image-cluster held-out split 上只达到 GateRecall=98.68%，
  detector-call reduction=2.45%，未达到约 99% 的验收目标，因此生产默认关闭。
- 方向词采用一次性 coarse hint：GeoRSCLIP 首轮 3x3 评价启用空间先验
  (`w_spatial=0.8`)，后续细分深度关闭该项，仅保留语义检索和父区域连续性；
  `spatial_prefilter=false` 确保不会因方向解析错误而硬裁掉整片图像。
- 六张 XLRS 大图已分别跑通纯类别 query 和类别+方位 query，共 12 次真实链路。
  两轮均 6/6 返回多 ROI、全局坐标、trace 与可视化；最终代码重跑总耗时分别约
  132.4 s 和 120.2 s（当前本地 CPU 环境，时延受 Windows 主机负载影响）。
- 真实输出 provenance 为 `georsclip / GeoRSCLIP-ViT-B-32`；checkpoint 加载为
  0 missing keys、0 unexpected keys。
- 当前 Python 环境不可见 CUDA，因此本地 peak VRAM 为 N/A。云端分层 runner 已
  自动记录 GPU 型号、CUDA、P50/P95 latency 和 peak allocated VRAM；生产 profile
  默认 `device=cuda`。数值必须在目标 GPU 主机生成，不能以 CPU 数据冒充。

## 生产运行

```powershell
$env:GEORSCLIP_CHECKPOINT = 'D:\models\GeoRSCLIP\GeoRSCLIP-ViT-B-32.pt'
python scripts/locator/run_uhr_locator.py `
  --config configs/locator/uhr_hierarchical.georsclip.yaml `
  --image D:\path\image.png `
  --question "Where is the harbor?" `
  --output reports\locator.json `
  --export-crops reports\locator_crops `
  --export-debug-overlay reports\locator_overlay.png
```

TaskGraph 使用同一 checkpoint 环境变量：

```powershell
python -m sat_rs_vlm.taskgraph.run `
  --provider-config configs/taskgraph/runtime.real.example.yaml
```

## 当前边界

- VRSBench 目标偏大，不能代替 XLRS/MME RealWorld RS 的带 GT 定量验收。
- 六图 smoke 证明链路和坐标可用，但没有 GT，不能声称真实 UHR Recall。
- Count gate 保持实验性和默认关闭。
- GPU latency、吞吐与 peak VRAM 仍需在能暴露 CUDA 的目标主机补测。
