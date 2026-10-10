# LAE-DINO 单 worker 与真实批处理

日期：2026-10-10。基础提交：`8ee1c788ba9c4ebcb17d943bd956f9a455dfc209`。

## 已实施

生产配置改为一个 LAE worker，默认 batch=16，允许配置 24、32 或更大。
这次保持独立 LAE 环境，没有合并模型环境、安装依赖或改变精度。

原来的五 worker 各自加载模型和 CUDA context；而 vendored
`inference_detector(model, list)` 会循环执行 batch=1，并不提供真正批量前向。
现在沿用原始 test pipeline，通过一次 `model.test_step` 处理一批 inputs/data_samples。
原来的缩放、BGR/RGB 转换、归一化及 bbox rescale 由原 pipeline/data preprocessor
继续负责，没有通过降采样或删候选换取显存。

LOCATE 切片与 counting 的 `detect_many` 均已接入批处理。
CPU 最多预制下一批切片；GPU 推理由一个消费者串行提交。
COUNT 每张裁剪图在源生成器继续推进之前就写入 BMP，防止读取已关闭的 PIL 图像。

同一主进程中，模型权重、模型配置、BERT、worker 程序、解释器、设备、推理模式、
batch 上限及 OOM 策略等身份一致的 provider 共用一个 sidecar。
LOCATE 与 counting 的 score_threshold、NMS、top-k 在每次请求中分别传入，
不参与共享身份，保持两者原有策略。
各 provider 关闭时采用引用计数，只有最后一个 owner 关闭时才释放进程。
不同权重或不同 runtime 身份仍分别使用各自的 worker。

## OOM、正确性和资源

- batch=16 OOM 时，依次尝试 8、4、2、1；重试相同切片，不漏掉剩余图像。
- worker 记录较小的安全上限，后续批次不再重复尝试此前失败的大 batch。
- batch=1 仍然失败或遇到非 CUDA OOM 错误时，返回失败，不能当作零目标。
- `batch_oom_retry: false` 可关闭自动重试。
- chunked-text 配置在 LAE 源码中仅允许 batch=1；显式降为串行并记录原因。
- 批协议校验外层及每张图像 ID，拒绝重复/未知/缺失/失败结果，按请求顺序还原。
- 每图坐标校验、原图映射、全局 NMS、候选截断和完整性标记沿用原逻辑。
- 检测完成后仍执行原有 worker 资源释放；OOM 重试之前也释放缓存。
- GPU 激活与临时 workspace 仍会随 batch 增大。OOM 自动拆分是事后处理，
  不等同于为后续 VLM 请求预留显存，也不能保证真实系统的峰值低于旧版本。
- batch=16 且 CPU prefetch 开启时，单次调用最多保留 32 个 BMP。
  1333×1333 RGB BMP 合计约 163 MiB；关闭 prefetch 后约 81 MiB。
  这只是传输文件大小，还需计入整图、CPU pipeline 张量和系统文件缓存。
  多个并发调用分别占用 CPU/磁盘缓冲，GPU 共享 worker 仍串行执行。
- 临时文件存于系统临时目录，批次消费后删除；正常完成、提前退出和异常均清理。

用户提供的每 worker 2.5–3.2 GB 估计可说明重复实例的代价，但不能将其直接相乘
推导安全 batch：其中包含激活峰值，batch 增大后这部分还会增长。
本次默认 16 是可回退的起点，不是已测得的最优值。

## 配置与 trace

在 `configs/taskgraph/runtime.real.example.yaml` 中同时设置：

```yaml
providers:
  detection:
    base_config:
      parallel_workers: 1
      parallel_max_workers: 1
      batch_size: 16
      batch_oom_retry: true
  counting:
    detector:
      batch_size: 16
      batch_oom_retry: true
```

两处 batch/OOM 配置应一致才能共享该 worker 身份。
旧的 LAE `parallel_workers: 5` 仍能加载，但实际 LAE 推理固定使用一个 worker。
没有批接口的其他 provider 保留原来的执行方式。

每个结果记录 `effective_batch_size`、`batch_size_requested`、`batch_limit`、
`batch_oom_retries`、`batch_worker_ms`、`worker_pid`。
`latency_ms` 为该协议批次总时间按图像数量平分，包含 OOM 重试，不能解释为单图独立耗时。
切片 trace 记录批次数及候选完整性；COUNT `call_count` 继续统计切片数量。

## 修改文件与原因

| 文件 | 修改 |
| --- | --- |
| `scripts/integrations/lae_dino_worker.py` | 真正批量前向、OOM 拆分、每请求后处理参数、initialize 协议 |
| `src/sat_rs_vlm/integrations/detectors/lae_dino_sidecar.py` | 单 worker、模型实例复用、批协议 ID 校验与请求分块 |
| `src/sat_rs_vlm/integrations/detectors/tile_transport.py` | 有界 CPU 批预取与异常/提前退出清理 |
| `src/sat_rs_vlm/integrations/detectors/tiled.py` | LOCATE 切片提交批次，保留坐标、排序和 NMS |
| `src/sat_rs_vlm/integrations/detectors/registry.py` | 将 batch 和 OOM 配置传给 LAE provider |
| `src/sat_rs_vlm/integrations/detectors/protocol.py` | 声明可选批接口及每图结果必须完整返回的契约 |
| `src/sat_rs_vlm/integrations/counting/detector_bridge.py` | counting 批处理、按连续 query/尺寸分批及原图坐标映射 |
| `configs/taskgraph/runtime.real.example.yaml` | 单 worker / batch=16，移除重复的 parallel_workers YAML 键 |
| `configs/locator/uhr_hierarchical.yaml` | 同步默认 batch 配置 |
| `tests/unit/integrations/test_dino_batching.py` | 新批处理、OOM、协议、共享实例、坐标及资源边界测试 |
| `tests/unit/integrations/test_detector_providers.py` | 多请求回归改为验证单进程串行处理 |
| `tests/unit/integrations/test_cpu_tile_prefetch.py` | 实际 COUNT executor 的串行/批量桥接与图像生命周期测试 |
| `docs/taskgraph_runtime.md` | 更新执行与 CPU/磁盘预算说明 |
| `environments/lae_dino/README.md` | 批协议和环境兼容说明 |

本轮未修改 HTTP/API、TaskGraph 规划、SELECT、模型权重或 Planner 训练数据。
工作区已有的其他改动保留在原处。

## 已执行验证

所有 Python 命令使用：
`C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe`。

1. 新批处理测试最初运行：32 passed，1.50s。
2. 原 detector providers、parallelism、CPU prefetch 回归：36 passed，1.96s。
3. counting integration、LOCATE routing v2、LOCATE/SELECT v2、UHR locator：
   93 passed，9.19s。
4. 补充逐图尺寸、后处理参数隔离、实际 COUNT executor 批处理后，
   新批处理与 CPU prefetch 最终运行：44 passed，1.56s。
5. 补齐可选批接口的类型声明后，对上述八个测试文件合并运行：
   **165 passed，11.50s**。

以上去重覆盖 165 项测试；第 4 次运行包含第 1/2 次中的部分测试。
最终运行记录为 `pytest.log` 和 `pytest.xml`（与本报告同目录）。
单元测试使用模拟 detector/轻量独立 JSONL worker；未加载真实 LAE 模型。
实际 `model.test_step` 批接口以 stub 验证一次调用接收 batch=1/8/16/32，
不等同于这些 batch 在完整模型上都能运行，也不表示准确率或速度提升。

命令（每组单独执行）：

```powershell
& 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe' -m pytest tests/unit/integrations/test_detector_providers.py tests/unit/integrations/test_detector_parallelism.py tests/unit/integrations/test_cpu_tile_prefetch.py -q --basetemp=.tmp/pytest-dino-regression
& 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe' -m pytest tests/unit/taskgraph/test_counting_integration.py tests/unit/taskgraph/test_locate_routing_v2.py tests/unit/taskgraph/test_locate_select_v2.py tests/unit/integrations/test_uhr_locator.py -q --basetemp=.tmp/pytest-dino-integration
& 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe' -m pytest tests/unit/integrations/test_cpu_tile_prefetch.py tests/unit/integrations/test_dino_batching.py -q --basetemp=.tmp/pytest-dino-final
```

修改文件的 Ruff 检查通过，`git diff --check` 通过。
5 个 detector/bridge/protocol 源文件的定向 Mypy 检查通过
（`--follow-imports=silent --ignore-missing-imports`，不是整个仓库的类型检查）；
初次检查发现的类型声明问题均已修复。详见同目录 `ruff.log`、`mypy.log`。

## 尚待真实模型实验

本机主 Python 缺少 LAE 所需的 mmcv/mmengine/mmdet，且本机是 8GB GPU，
不能代表此前可运行五 worker 的云端环境。本轮未改造/安装模型环境，也未在云端部署。
因此没有真实 UHR 图片的前后耗时、完整模型批量兼容性、准确率对比或 GPU 峰值数据。

后续应在目标 GPU 和已有 LAE 环境中，用用户指定的
`D:\Desktop\tzb-2026\results\uhr_locator` 图片比较 batch=1/8/16/24/32，
逐次记录有效 batch、OOM 次数、切片吞吐、端到端时间及整体进程显存峰值。
还需比较逐图 bbox/score 与计数结果，观察 fp32 批量内核数值差异及阈值边界。
一次试验只保留一个候选批配置，检查后续 VLM 实际运行时的显存余量。
预计超过 30 秒的真实实验须遵守本仓库后台任务规则。
