# 推理性能监测

`scripts/evaluate_rs_vlm.py` 默认在真实模型生成流程中启用性能监测，并在评测输出目录生成 `performance_report.json`。该文件用于答疑要求的资源与时延申报，和任务质量评测的 `summary.json`、逐样本结果 `predictions.jsonl` 配套保存。

## 运行方式

```powershell
python scripts/evaluate_rs_vlm.py `
  --config configs/eval/qwen3vl_eval.yaml `
  --checkpoint <checkpoint目录> `
  --output-dir <评测输出目录>
```

默认读取配置中的：

```yaml
performance:
  enabled: true
  batch_size: 1
  repeats: 1
  warmup_samples: 2
  continue_on_error: true
```

也可临时关闭或改变预热数量：

```powershell
python scripts/evaluate_rs_vlm.py --config <配置文件> --no-performance-monitor
python scripts/evaluate_rs_vlm.py --config <配置文件> --warmup-samples 5
```

## 自动记录内容

| 类别 | 字段 | 含义 |
| --- | --- | --- |
| 端到端延迟 | `latency_ms.mean/p50/p95/min/max` | 单样本从图像预处理/拼接开始，到文本解码完成且CUDA同步结束的时间，单位ms。 |
| 首Token时间 | `ttft_ms` | 与端到端延迟使用同一起点，到生成循环报告首个Token的时间。若当前推理框架不支持回调则样本数为0，不伪造数值。 |
| 吞吐 | `generation_tokens_per_second` | 生成Token数除以`generate`阶段总耗时。 |
| 解码速度 | `decode_tokens_per_second` | `(生成Token数-1)/(首Token后至生成结束时间)`；只有至少两个Token时统计。 |
| 显存 | `memory_mb.gpu.peak_allocated_mb` / `peak_reserved_mb` | 在预热后重置CUDA峰值统计，再记录正式评测阶段的峰值分配/保留显存。 |
| CPU内存 | `inference_peak_process_rss_mb` | 正式评测阶段多次采样到的进程工作集峰值。`os_process_peak_rss_mb`是操作系统可见的进程生命周期峰值，应与前者区分。 |
| 运行完整性 | `requested_samples/completed_samples/failed_samples/warmup_samples` | 计划、完成、失败及未计入统计的预热样本数量。 |
| 启动 | `startup_and_model_load_ms` / `model_load_ms` | 依赖导入和模型加载总时间、以及单独模型/处理器加载时间。 |
| 环境 | `environment` | 操作系统、Python、CPU逻辑核数、GPU、CUDA、关键依赖版本及执行精度/设备配置。 |
| 驱动 | `environment.accelerator.nvidia_smi` | 自动查询GPU型号、驱动版本和显存总量；本机无`nvidia-smi`时保留不可用原因。 |
| 模型资源 | `model_resources` | 已加载模型逻辑参数量，以及本机可访问的基础模型/Adapter目录实际存储大小。 |
| 输入规格 | `input_profile` | 图像数量、原图尺寸、Processor后视觉Tensor形状、`image_grid_thw`、数据元信息声明的切片数及可解析的视觉Token数。 |

逐样本的`predictions.jsonl`保留原有`inference_latency_ms`，并新增`performance`对象，其中包含端到端延迟、TTFT、生成耗时、输出Token数、Token速度和实际视觉输入规格。若Processor不能可靠导出视觉Token数，脚本会写入`null`及状态原因，不猜测数值。若启用了旧的变化检测辅助二分类路径，`performance.system_end_to_end_latency_ms`会将该辅助推理时间一并计入。

## 对比规范

1. 当前入口固定`batch_size=1`、`repeats=1`，以单样本端到端时延为口径。配置为其他值会明确失败，防止误报性能数据。
2. 比较模型、量化或容错方案时，应保持硬件、软件版本、样本顺序、批大小、预热次数、生成参数、图像输入规格和缓存策略一致。
3. 延迟报告仅比较同一`measurement_scope`下的结果；当前评测入口固定为`single_sample_end_to_end`。
4. 不将加载时间混入稳态P50/P95，但应单独报告冷启动/模型加载时间。
5. 结果文件应保留`performance_report.json`、`summary.json`和原始`predictions.jsonl`，以便复核每个统计量的样本数与失败情况。
