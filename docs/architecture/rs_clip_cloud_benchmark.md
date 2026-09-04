# 五模型 RS-CLIP 云端分层评测

本说明随 `rs_clip_cloud_benchmark_bundle.zip` 一起交付。解压后进入
`rs_clip_cloud_benchmark/` 目录再执行下列命令；压缩包不包含数据集或模型权重。

## 当前结论

队长已确定 **GeoRSCLIP ViT-B/32** 为 Region Retriever 的生产默认模型。
RemoteCLIP、FarSLIP、SatelliteCLIP 和 Git-RSCLIP 仅保留为公平对比 provider，
不会根据这份云端排名自动改写生产默认配置。全局坐标 provenance、多候选、缓存、
Count gate、benchmark 和 overlay 已经搭建完成。

云端实验只用于审计五个模型在更大、独立数据集和真实 GPU 上的表现是否稳定，
以及补测 GPU latency、吞吐和 peak VRAM。它不会自动覆盖 GeoRSCLIP 的工程定版。

## 固定模型

1. RemoteCLIP ViT-B/32
2. GeoRSCLIP ViT-B/32
3. FarSLIP ViT-B/32
4. SatelliteCLIP
5. Git-RSCLIP-base

五个模型必须使用同一 staged manifest、3x3 candidates、Top-5、category query 和
coverage threshold=0.5。runner 每次只加载一个模型，防止五个模型同时占用显存。

## 数据清单

云端数据转换为 JSONL，每行格式如下：

```json
{"id":"sample-0001","image":"images/0001.png","query":"the airport near the coast","category":"airport","gt_boxes":[[120,80,640,510]]}
```

要求：

- `id` 全局唯一；
- `image` 可以相对 `RS_CLIP_DATA_ROOT`，不写本机盘符；
- `gt_boxes` 必须是原图 absolute pixel `xyxy`；
- normalized 坐标会在 preflight 阶段被拒绝；
- 所有 bbox 必须有限、非退化且位于图像范围内。

## 云端环境变量

```bash
export RS_CLIP_MANIFEST=/root/data/rs_clip_large/manifest.jsonl
export RS_CLIP_DATA_ROOT=/root/data/rs_clip_large
export RS_CLIP_OUTPUT_ROOT=/root/autodl-tmp/rs_clip_results

export REMOTECLIP_CHECKPOINT=/root/models/RemoteCLIP-ViT-B-32.pt
export GEORSCLIP_CHECKPOINT=/root/models/GeoRSCLIP-ViT-B-32.pt
export FARSLIP_CHECKPOINT=/root/models/FarSLIP1_ViT-B-32.pt
export SATELLITECLIP_MODEL_PATH=/root/models/SatelliteCLIP
export GIT_RSCLIP_MODEL_PATH=/root/models/Git-RSCLIP-base
```

安装并验证 CUDA：

```bash
bash scripts/cloud/setup_rs_clip_benchmark.sh
```

如果云端拿到的是原始 VRSBench 标注，先生成统一清单（不要设置 `--limit`，分层由
runner 自己完成）：

```bash
python scripts/make_vrsbench_retriever_manifest.py \
  --annotation-dir /root/data/VRSBench/referring_json \
  --image-dir /root/data/VRSBench/images \
  --output /root/data/rs_clip_large/manifest.jsonl
```

请按云端实际目录修改参数。转换器会把 VRSBench normalized xyxy 修正成原图像素
坐标；随后 preflight 会再次检查图片、类别、重复 id 和 bbox。

不要直接上传本机旧的 `vrsbench_retriever_val_200.jsonl`：该文件包含坐标修复前的
normalized bbox，preflight 会按设计拒绝。为避免 Windows 绝对路径和旧标注污染
云端实验，应在云端原始数据目录中重新运行上述转换命令，并让
`RS_CLIP_MANIFEST` 指向新生成的文件。

## 分层运行

先只验证清单、图像和 bbox，不加载模型：

```bash
python scripts/cloud/run_rs_clip_benchmark.py \
  --config configs/cloud/rs_clip_benchmark.yaml \
  --tier smoke50 --validate-only
```

再做完整 dry-run，检查五个模型路径、CUDA、固定协议、样本数和 SHA-256，但不加载
权重、不做推理：

```bash
python scripts/cloud/run_rs_clip_benchmark.py \
  --config configs/cloud/rs_clip_benchmark.yaml \
  --tier smoke50 --dry-run
```

第一层必须让五个模型各跑同一批 50 条：

```bash
bash scripts/cloud/run_rs_clip_tier.sh smoke50
```

检查以下文件生成后再进入下一层：

```text
${RS_CLIP_OUTPUT_ROOT}/smoke50/tier_status.json
${RS_CLIP_OUTPUT_ROOT}/smoke50/ranking.md
${RS_CLIP_OUTPUT_ROOT}/smoke50/models/<model>/report.json
```

然后逐层运行，不能跳级：

```bash
bash scripts/cloud/run_rs_clip_tier.sh standard500
bash scripts/cloud/run_rs_clip_tier.sh large2000
bash scripts/cloud/run_rs_clip_tier.sh full
```

`standard500` 会检查 `smoke50` 是否五模型全部完成；`large2000` 检查
`standard500`；`full` 检查 `large2000`。runner 不会自动连续运行下一层。

## 中断续跑和单模型调试

每条结果立即追加到：

```text
<tier>/models/<model>/rows.jsonl
```

同一命令重启后会校验样本顺序，并从最后一条继续。单模型调试：

```bash
python scripts/cloud/run_rs_clip_benchmark.py \
  --config configs/cloud/rs_clip_benchmark.yaml \
  --tier smoke50 --only-model remoteclip
```

单模型完成不会把整个 tier 标记为完成。必须五模型全部执行，才生成
`tier_status.json` 和最终排名。

## 每层输出

- 每模型逐条 JSONL 和 CSV；
- 完整 benchmark report；
- Recall@1/3/5、MRR、AP、NDCG@5、GT coverage、selected area ratio；
- 排除前三条 warm-up 后的 GPU P50/P95 latency；
- peak allocated VRAM；
- GPU、CUDA、PyTorch、manifest SHA-256；
- 五模型 `ranking.json`、`ranking.csv`、`ranking.md`。

前 50 条只用于检查权重、依赖、显存、坐标和结果格式。是否扩大到 500 条，重点看：

- 五模型全部完成且没有 bbox/data error；
- GPU 无 OOM；
- 每个模型结果数都是 50；
- 所有报告的 staged manifest SHA-256 相同；
- Recall/coverage 不是明显异常值。
