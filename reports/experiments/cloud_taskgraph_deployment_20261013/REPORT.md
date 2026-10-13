# 完整 TaskGraph 云端部署验证，2026-10-13

更新：用户随后更正目标 Planner 日期为 **2026-10-11**。已核对该 adapter 的云端 SHA256
`d7169cce768dcd01e86f1c2fd2847f7fddf85e45f9d61db3e23180a97211a4e6`，并切换主服务至 PID `30837`。
现有 HTTPS 隧道 PID `30076` 保留，原始部署的 PID `29050` 已正常退出。
以下五类冒烟耗时是切换前的历史记录；后续 25 条 UHR 原题使用 10 月 11 日 Planner。
后续结果见同目录 `UHR25_REPORT.md`。

## 部署结果

在用户指定的 AutoDL 机器上，完整生产入口
`sat_rs_vlm.competition.protocol_v2.server:create_app --factory`
已加载模型并完成真实 HTTP 推理。复用既有 Protocol 2.0、TaskGraph、模型工厂和算子，
未另建单模型服务或并行推理架构，未训练或修改任何权重。

部署目录为 `/root/sat-rs-vlm-deploy-aee1e1d`，基础源码提交为
`aee1e1d7f43c474fab9ec9399d0b60792e203316`，叠加本次启动、日志与兼容性修复。
没有覆盖 `/root/autodl-tmp/sat-rs-vlm` 中的已有工作树，也没有停止其他训练/评测进程。

运行目录：`/root/autodl-fs/experiments/sat-rs-vlm_protocol_v2_20261013`。
服务 PID `29050`，LAE worker PID `29351`，HTTPS 隧道 PID `30076`。
内部监听 `127.0.0.1:8000`，一个 Uvicorn worker；已有请求锁串行处理推理。
公网地址与密钥另存运行目录中的 `endpoint.txt`、`api-key.txt`，不写入 Git。
本地接入文件位于 `.tmp/cloud-taskgraph-service-20261013/`。
云端题目采集已启用，文件为运行目录中的 `questions.jsonl`，权限 0600。

原 4B 诊断服务已有可用工具
`/root/autodl-fs/experiments/competition_4b_llm_only_20261013_153508_6fae/bin/cloudflared-local`。
本次上传的 Linux 工具与它相同，版本 2026.10.0，SHA256
`d33ff2d14475178d2012c2c56beba87389ac5ded27649519f198a7d3134a99db`。
原目录中的 `bin/cloudflared` 是空文件，不应使用。
用户指出现有工具时，新隧道已在中断前成功启动，随后保留单个活动隧道，未重复启动。
临时 URL 随隧道进程生命周期变化，不等同于固定生产域名。

## 修复范围

- `scripts/competition/start_cloud_protocol_v2.sh`：后台启动完整生产服务，显式传递
  counting system 源目录；恢复调用者环境变量，避免旧 `/root/autodl_env.sh` 覆盖部署路径；
  切换到部署仓库根目录，保证相对 ontology/config 路径正确；拒绝占用端口与覆盖运行目录。
- `configs/competition/uvicorn_logging.yaml`：启用协议安全元数据日志，记录实际路由和耗时。
- `instance_postprocess.py`：修复 Python 3.8 导入时类型别名求值崩溃。
  这是实际 LAE worker 的阻塞点，不是模型推理算法变化。
- 新增生产装配无模型测试、真实隔离 worker 解释器导入回归；更新部署文档。

首次真实计数虽然返回 HTTP 200，但 worker 导入失败，答案为零，不能当作链路成功。
修复类型别名后，在不重启主服务的情况下，既有 sidecar 客户端重新拉起已退出的 worker，
检测器成功加载并完成真实请求。后续 worker 日志未出现新的 traceback。

测试辅助脚本曾错误调用 `compatibility_report` 属性为方法、误把 JPEG 定位资产声明成 PNG。
这两项是测试脚本错误，已修正并重跑；服务正确拒绝 MIME 不一致请求。
失败记录保留在本次运行日志，最终通过结果单独保存，未把失败计入成功数。

## 模型与 Planner 版本边界

4B 底座：`/root/autodl-tmp/models/Qwen3-VL-4B-Instruct`。
2B 底座：`/root/autodl-tmp/models/Qwen3-VL-2B-Instruct`。

首次部署时的 Planner adapter（后续已替换）：
`/root/autodl-tmp/outputs/taskgraph/qwen3vl_4b_planner_hard_refine_v1_20260902_234235`。
`adapter_model.safetensors` 大小 132195448 字节，mtime 为 2026-09-03 00:05:05 +0800，SHA256：
`defad37c62a53cb928d8879d8f690cc55806dffee7a81a547f121c9975511570`。
这是已有老 Planner 路径，**不能称为已验证的 10 月 10 日冻结版本**。
用户更正目标日期后，实际服务已改用
`/root/autodl-fs/data/sat-rs-vlm-training/planner_4b_clean_20261011/output/training`，
权重大小同为 132195448 字节，mtime 为 2026-10-11 14:30:26 +0800，SHA256 见本文更新。
启动器要求显式指定 Planner adapter，不会把这个旧目录自动固定为所有部署的默认值。

双时相 adapter：
`/root/autodl-fs/experiments/sat-rs-vlm_cloud_eval_20261012_125359/changechat_4b_continuation_retry3/runs/llm_only`。
2B adapter：`/root/autodl-tmp/outputs/vrsbench_levircc_replay_formal/round_2_adapter`。
GeoRSCLIP：`/root/autodl-fs/models/GeoRSCLIP/GeoRSCLIP-ViT-B-32.pt`。
LAE-DINO：`/root/autodl-fs/rs_detectors/lae_dino/checkpoints/lae_dino_swint_lae1m-28ca3a15.pth`。
LAE worker 使用 `/root/miniconda3/envs/lae-dino-gpu/bin/python3`，实测 Python 3.8.20。
主服务使用 `/root/miniconda3/envs/rs-vlm/bin/python3.12`。

## 验证结果

- 云端无模型生产装配及协议/UHR 测试：101 passed，0 failed。
- 云端真实 Python 3.8 worker 导入及实例后处理测试：13 passed，0 failed。
- 本地指定 WindowsApps Python 回归：78 passed，4 skipped，0 failed。
  跳过为本地未配置的 LAE 隔离解释器、两项云端环境测试、Windows 符号链接条件。
- 未声称全仓库全部测试通过。单模型诊断服务已有的健康检查测试问题不属于本次完整入口。
- 五类真实模型请求均 HTTP 200，最终答案满足现有 `check_response` 验证。
- GeoRSCLIP 独立真实评分：两个区域均得到有限值；全部 151277313 个参数加载，
  missing/unexpected keys 均为零，compatibility_status 为 compatible。
- 本机经公网 HTTPS：鉴权健康检查 200、未带密钥 401；原始开发定位题推理 200，
  像素坐标及响应回显验证通过。没有在比赛网站代替用户提交或消耗官方评测次数。

| 请求 | 实际路由 | 最终实测耗时 | 说明 |
| --- | --- | ---: | --- |
| 双时相 RSU_D_0058 | DIRECT_VLM / 4B | 0.94 s | 原始图片对和九区域约束，返回 top_right |
| 整图船只计数 | TASKGRAPH_UHR | 3.67 s | 热请求，返回 0；首次成功冷请求 21.84 s |
| 低分辨率选择题 | DIRECT_VLM / 2B | 0.46 s | 返回候选 C |
| 低分辨率短文本 | DIRECT_VLM / 2B | 0.43 s | 返回非空短文本 |
| RSU_B_0035 像素 BBOX | TASKGRAPH_UHR | 9.17 s | 原始开发题，合法 xyxy，未使用直接 VLM 兜底 |
| 同题通过公网 HTTPS | TASKGRAPH_UHR | 8.26 s | 服务器内部 7.26 s，返回相同像素 BBOX |

这些是少量功能冒烟结果，缓存冷热不同，**不代表准确率或正式全量时延分布**。
没有将协议合法性当作视觉答案正确性，也没有用单题耗时承诺所有正式题预算。

共享 4B：生产装配确认 Planner 与 ChangeChat 使用同一个 manager，真实请求中
切换计数从 0 增至 2，并可继续多轮切换。没有加载第二份 4B 底座。
未做关闭共享后的显存对照，不能报告节省显存的实测差值。
最终一轮整卡采样峰值 23204 MiB，前一轮最高 23344 MiB（约 22.8 GiB / 24 GiB）；
最后空闲驻留为主服务 13928 MiB、LAE worker 3938 MiB。
采样间隔 0.5 秒，可能漏掉短暂峰值，且独立 CLIP 组件测试也计入整卡采样。

## 仍需关注

- 用户更正的 10 月 11 日 Planner 来源已匹配既有训练、下载和评测记录；
  不再把不存在证据的 10 月 10 日版本作为本次部署目标。
- 24 GB 显存余量较小，保持单请求锁、一个 worker，不提高模型/语义候选并发。
- 候选验证计数器按当前既有策略在部分检测异常时仍可输出零值；本次修复的是实际
  worker 崩溃，未更改这项计数回退产品策略。真实部署故障排查必须看 worker 日志。
- 正式 800 图片包未在本次替换，当前 dataset_id 为 rsu-dev100-v2。
  切换正式包时应同步比赛端 dataset_id 与服务配置。
- 尚未执行官方平台 check_api/正式 8 题或 100 题评分；临时隧道不提供固定域名保证。
- 未修改 Planner 清洗、训练代码或其他本地未提交工作。

## 状态与停止命令

云端查看状态：

```bash
ps -p 30837,30076 -o pid,etime,comm
tail -n 30 /root/autodl-fs/experiments/sat-rs-vlm_protocol_v2_20261013/service_planner_20261011/server.stderr.log
```

需要主动停止时，由用户执行（本次没有执行）：

```bash
kill -TERM 30076 30837
```

主服务正常退出会调用既有 runtime/provider 关闭流程并回收 LAE sidecar。
日志、机器路径和 SHA256 是本次验证记录，PID 与临时 URL 不应作为长期静态配置。
