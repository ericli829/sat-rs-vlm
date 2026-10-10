# DINO 执行方式初查：单 worker batching 与主环境兼容

日期：2026-10-10。本轮按“考虑两种方案”进行源码检查和小型接口探测，未修改生产执行代码、配置、环境依赖或模型权重，也未启动完整模型性能测试。

## 结论

建议先实现 **单个 DINO 常驻进程、单模型实例、真正的 batch 前向**，保留 CPU 裁图预取；随后在独立实验环境中验证 **同进程兼容**。两者可以结合，并非互斥方案。

当前较大的资源浪费来自并发 worker 的模型副本，不是所有多线程都应移除。仅共用 Python 环境而继续用子进程，不能自动共享模型权重或 CUDA 上下文。相对于“单独一个 DINO 子进程”，改成同进程可能节省独立上下文与缓冲开销，但节省多少尚未实测。

## 1. 当前执行链

```text
TiledProposalProvider.predict
  → N 个 tiled-detector 调度线程
  → LAEDinoSidecarProvider 的 N 个 client
  → N 个独立 Python 子进程
  → 每个子进程 _load_detector 各加载一个 DINO
  → 每次请求一张 tile，inference_detector(...)
```

证据：

- `configs/taskgraph/runtime.real.example.yaml:82` 和 `:86` 重复声明 parallel_workers，后一个值为 5；常用 YAML loader 后值覆盖前值。parallel_max_workers=5。实际部署若使用别的配置，必须另核对解析后的配置和实际 PID；本轮没有读取云端活动配置。
- `src/sat_rs_vlm/integrations/detectors/registry.py:41` 起把外层并发参数传给基础 provider。
- `tiled.py:327` 创建调度线程池；`lae_dino_sidecar.py:290` 起按 worker 数构造独立 client；client.start 用 subprocess.Popen 启动 worker。
- `scripts/integrations/lae_dino_worker.py` 每个进程在主循环之前加载模型，每次请求调用一次 `_predict`。
- 显式 worker 数只受 parallel_max_workers 限制，只有 auto 分支才根据空闲显存估算；它也不是根据全系统实测峰值的动态 batch 调度器。

另有两处需要区分：

1. `tile_transport.py` 的 tile-cpu 线程只预取下一张 CPU 裁图/文件，不加载另一份 DINO，通常应该保留。
2. `CountingProposalDetectorBridge.detect_many` 目前已经串行消费 tile；它是 CPU 预取加逐张推理，**不是真 batch**。运行时分别构造 LOCATE detection provider 与 counting provider，二者可能分别加载同一 checkpoint。需要两条路径都使用后才会出现这类额外副本；`COUNT(EntitySet)` 不必重新检测。以后共享模型时必须保留各自的阈值、top-k、NMS 及结果完整性元数据，不能混用计数/定位策略。

## 2. 第二种方案不能只改 batch_size

当前 worker 协议只有单个 image，provider 只有单图 predict；没有 DINO 批量配置到运行时的完整调用链。

已读取本地固定版本 LAE-DINO `6b1519626e39d1f39f8ed1f38761c20f7e0e8c35` 的 `mmdetection_lae/mmdet/apis/inference.py`。其 inference_detector 虽接受图片列表，内部却在 for 循环里为每张图单独调用 `model.test_step`。

为确认这一点，从原文件提取该函数，使用 stub model/pipeline 执行；没有改写函数逻辑：

| 传入图片数 | test_step 调用次数 | 每次前向的输入数 |
| ---: | ---: | --- |
| 1 | 1 | 1 |
| 2 | 2 | 1、1 |
| 4 | 4 | 1、1、1、1 |

见 [list_api_probe.json](list_api_probe.json)，包含源文件 SHA256。该探测只证明调用结构，不代表真实模型吞吐测试。

实际实现需要：

- 增加 predict_batch / worker 批量协议，按同一目标文本及近似输入形状组织 tile；由一个 worker 顺序处理 batch。
- 保留原 test pipeline 的颜色空间、归一化、FixScaleResize、original shape、scale_factor；组装多项 inputs/data_samples，一次调用 test_step，让 data_preprocessor 完成 padding 与堆叠。
- 保留每项 request/tile ID、局部坐标、原图映射、顺序及候选完整性标记；不以提高吞吐为由减小图像分辨率或截断候选。
- LOCATE tiled 与 counting bridge 同步接入；后者不能因 detect_many 的名字而误认为已经支持 GPU batch。
- 从 batch=1、2 开始，显存有余量再试 4。按处理后像素数和全系统峰值约束预算，不承诺固定的大 batch；OOM 时分割原 batch 重试，不能漏图或少返回候选。
- LAE 的正常 predict 分支有 batch 维度，但 chunked text 分支明确 assert batch=1，需要保持串行或显式拒绝不支持的组合。不能仅凭正常分支源码就宣称所有 batch 模式可运行。

“单线程”在这里应指只有一个 GPU 推理消费者。CPU 裁图、解码预取和库内部并行并不需要一律强制为 1。

## 3. 第一种方案的兼容性证据与边界

本机指定解释器探测结果：

| 项目 | 当前本机 |
| --- | --- |
| Python | 用户指定 WindowsApps Python 3.11 |
| Torch / torchvision | 2.6.0+cu126 / 0.21.0+cu126 |
| Transformers | 5.13.0 |
| NumPy | 2.4.0 |
| MMCV / MMEngine / MMDetection | 未安装 |
| GPU | RTX 4060 Laptop，8188 MiB |

LAE 本地 README / lae_requirements.txt 的参考栈是 Python 3.8、Torch 1.10+cu113、Transformers 4.42.3、MMCV 2.2.0、MMEngine 0.10.4 和定制 MMDetection 3.3.0。这是源码记录的参考环境，**不是当前云端运行版本的实测记录，也不是已证明的唯一可用版本**。

兼容实验需要同时检查：

1. 目标机器 Torch/CUDA 对应的 MMCV 扩展能否导入及执行，尤其模型用到的 deformable attention；pip 安装成功不算通过。
2. 确实加载本项目固定的定制 mmdet/LAEDINO，而不是用发行版 GroundingDINO 替换权重/结构。
3. 当前 Transformers 的 BERT、tokenizer、自定义 mask 以及 LAE 相关 helper 是否保持原行为。不能为迁就 detector 随意降级整个 VLM 环境。
4. 同进程 LAE 初始化会修改 sys.path、MMEngine registry/default scope、离线环境变量；需处理进程级副作用。显存释放也要避免干扰其他模型缓存。

已做一个有限的实际 API 探测：使用当前 Torch/Transformers 创建小型随机初始化 CPU BERT，测试 batch=1/2、二维/三维 attention mask、默认 SDPA/显式 eager，共 8 个组合全部运行成功。LAE 使用三维子句 mask，因此这个结果排除了“当前版本必然不支持这种 mask”的简单判断。

见 [bert_api_probe.json](bert_api_probe.json)、[probe_bert_api.py](probe_bert_api.py)。**没有加载正式 BERT 权重、MMCV、LAEDINO 或 VLM，不证明整套环境兼容、输出等价或显存更低。** 本机缺少检测依赖，本轮没有向工作环境安装/降级这些包。

## 4. 资源与性能判断

| 方案 | 预期收益 | 代价与待测内容 |
| --- | --- | --- |
| 单 worker + 真 batch | 减少多份权重与多个 CUDA 上下文；批量利用 GPU；保留现有环境隔离 | batch 激活与 padding 增加峰值；需新调用链与真实输出一致性检查 |
| 环境兼容但仍是子进程 | 部署依赖可统一 | 继续有独立模型/上下文，不能据此预期显著省显存 |
| 同进程 + 单 DINO 实例 | 可能减少独立上下文、缓冲和文件传输；可进一步与 batch 结合 | MMCV/Torch/HF/定制源码兼容与共享进程状态需要实测 |

不能把 5 个 worker 的速度直接当作单 worker batch=5 的速度；也不能把 batch 激活的增加忽略。主要驻留内存收益首先来自减少副本。同进程的额外收益应在“均为一份 DINO、同样 batch、其他模型驻留相同”的条件下对比。

其他发现：worker 当前每次请求后都会执行 gc.collect、cuda.empty_cache 和可用时 malloc_trim；这些操作可能增加同步/重分配开销，尚未计时。实现 batching 时宜比较每 batch 清理与按预算清理，不能在显存吃紧时无条件取消全部释放。

还确认了一项真实实验的前置问题：client.preload/initialize 发送 `operation=initialize`，但当前 worker.main 对所有请求都要求 image/target_phrase，没有处理该 operation。用原 worker.main 加 stub loader/predictor 探测，initialize 返回 `missing request field: image`，普通 predict 请求可派发；见 [preload_protocol_probe.json](preload_protocol_probe.json)。正式基线应先补齐初始化协议，否则可能把预加载失败误判成依赖不兼容。本轮只记录问题，没有修改该协议。

## 5. 建议实测矩阵

使用已授权的 `D:\Desktop\tzb-2026\results\uhr_locator` 原图，挑 2–3 张分别含密集 ship/vehicle 和稀疏目标的图片，先验证输出，再测速度：

1. 旧多 worker 配置作为基线，仅在目标机器显存允许时运行，不在本机 8GB 上盲目加载五份。
2. 隔离环境单 worker：batch=1、2；有实测余量后加 4。
3. 同进程实验环境：先 batch=1 验证完整 LAEDINO + Planner + 语义 VLM 共存，再与第 2 项用相同 batch 对照。

每组固定原图、切片、查询、阈值、精度和 NMS；关闭 proposal cache 对计时的影响，分别记录冷启动及热运行。检查候选数、框匹配/IoU、分数容差、计数结果、tile ID/坐标映射，以及 candidate_pool_complete。不要仅比较最后一个答案；batch padding 或浮点差异也需要记录。

性能记录包含：完整 wall time、解码裁图/BMP 写读、IPC、模型前向、CPU 后处理/NMS，以及 nvidia-smi 的全进程峰值和进程内 allocated/reserved。多 worker 的 base_latency_ms 是各 tile 耗时之和，不能当端到端耗时。显存预算必须包含其他驻留模型和后续 VLM KV/视觉峰值。

本轮没有执行上述真实权重/云端/GPU吞吐实验，没有据静态分析声称速度或准确率提升。真实长测试如超过 30 秒，应遵循本仓库后台运行规则；尚未启动后台任务。

## 6. 本轮验证与修改范围

- 只新增本目录的分析报告与探测产物，生产源码、默认配置及包环境均未修改。
- 原 inference_detector 列表调用结构探测：1/2/4 张均确认逐张前向。
- 小型随机 CPU BERT API 探测：8/8 成功。
- 探测脚本 Ruff check / format check 通过。
- 已解析示例 YAML，确认生效 worker 数为 5；依赖版本和相关源码 SHA256 见 [inspection.json](inspection.json)。
- worker 初始化协议探测复现一项已有失败，已单独记录，不伪称所有检查通过。
- 不把接口探测当作完整兼容性、VRAM、吞吐或准确率结论。
