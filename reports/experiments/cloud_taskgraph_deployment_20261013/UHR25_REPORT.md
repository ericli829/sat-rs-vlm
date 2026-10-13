# 开发集 25 条 UHR 原始请求实测

使用开发集原始 100 题记录中的全部 C 组 25 题，每题一张不同的 UHR 图片。问题、图片标识/hash、选项和答案约束保持原样，仅修改 request_id 以标识本次重放。

- 完成：25/25；HTTP/答案契约通过：25/25。
- 最终路由：TaskGraph 直接完成 10/25（40%），现有 2B VLM 兜底完成 15/25（60%）。
- 平均 14.64 s；中位数 13.46 s；P95（nearest rank）27.96 s；最长 35.95 s。
- 整卡显存采样峰值 24004 MiB，采样间隔 0.5 秒，可能漏掉瞬时峰值。
- 单个既有生产服务串行调用；未启动第二份模型推理进程。
- 当前批次经云端 loopback HTTP 调用，网络耗时不包含公网隧道；此前公网真实推理已单独通过。
- 没有参考答案，以上通过率是协议与服务可用率，不是模型准确率。DIRECT_VLM 表示高分辨率请求最终走现有 2B 兜底，不计作 TaskGraph 直接完成。

接口在本批次中可用，但 TaskGraph 的直接完成率仍偏低，不能据此宣称主链路已稳定。
TaskGraph 直接完成的 10 题平均 11.31 秒；兜底的 15 题平均 16.87 秒，后者包含前置
TaskGraph 尝试耗时，不能当作纯 2B 推理时间。25 题均未触发 HTTP 120 秒超时。
采样峰值约 23.44 GiB，总显存 24564 MiB，余量只有约 560 MiB；不宜提高推理并发。

现有 HTTP 元数据没有持久化每题的 DSL、失败算子或最终答案契约失败细节，
因此这次能够确认兜底发生及其比例，不能仅凭该批记录将 15 题都归因为 Planner、
检测器、答案解析或 OOM。服务端日志没有可见的新 traceback/OOM，但这也不能排除
已被内部捕获的异常。未为提高通过率而放宽解析或答案格式要求。

## Planner 与运行证据

- Planner：`/root/autodl-fs/data/sat-rs-vlm-training/planner_4b_clean_20261011/output/training`。
- 权重 SHA256：`d7169cce768dcd01e86f1c2fd2847f7fddf85e45f9d61db3e23180a97211a4e6`。
- 原始请求清单 SHA256：`9f87f0ffa5c3e889a2e9476c83c3cc6b4b28027d7c321a9d0a730c1d94d13a6e`。
- 服务 PID：`30837`；运行目录：`/root/autodl-fs/experiments/sat-rs-vlm_protocol_v2_20261013/service_planner_20261011`。

权重与 10 月 11 日清洗训练、下载和评测记录一致。未使用 10 月 13 日续训产物。HTTPS 隧道与接入密钥保持原样，服务继续运行。
批次结束后，本机再次通过现有公网 HTTPS 健康检查，HTTP 200、状态 ready。

## 逐题结果

| ID | 答案 | HTTP | 最终路由 | 耗时(s) | 显存采样峰值(MiB) |
| --- | --- | ---: | --- | ---: | ---: |
| RSU_C_0003 | B | 200 | TASKGRAPH_UHR | 24.85 | 18700 |
| RSU_C_0009 | C | 200 | TASKGRAPH_UHR | 13.52 | 20114 |
| RSU_C_0055 | A | 200 | DIRECT_VLM | 17.97 | 20110 |
| RSU_C_0076 | A | 200 | TASKGRAPH_UHR | 16.06 | 20116 |
| RSU_C_0080 | B | 200 | DIRECT_VLM | 17.52 | 22870 |
| RSU_C_0082 | D | 200 | DIRECT_VLM | 13.46 | 23612 |
| RSU_C_0104 | A | 200 | DIRECT_VLM | 11.23 | 23818 |
| RSU_C_0121 | C | 200 | TASKGRAPH_UHR | 13.63 | 21670 |
| RSU_C_0142 | A | 200 | DIRECT_VLM | 27.96 | 23870 |
| RSU_C_0146 | A | 200 | DIRECT_VLM | 16.22 | 23868 |
| RSU_C_0189 | A | 200 | TASKGRAPH_UHR | 3.96 | 23610 |
| RSU_C_0209 | A | 200 | TASKGRAPH_UHR | 6.08 | 22088 |
| RSU_C_0232 | B | 200 | DIRECT_VLM | 11.43 | 22792 |
| RSU_C_0249 | A | 200 | DIRECT_VLM | 8.69 | 22850 |
| RSU_C_0251 | A | 200 | TASKGRAPH_UHR | 7.25 | 23736 |
| RSU_C_0262 | C | 200 | DIRECT_VLM | 21.69 | 24004 |
| RSU_C_0263 | D | 200 | DIRECT_VLM | 13.00 | 23150 |
| RSU_C_0285 | B | 200 | DIRECT_VLM | 20.59 | 23876 |
| RSU_C_0289 | B | 200 | DIRECT_VLM | 14.03 | 23890 |
| RSU_C_0305 | B | 200 | TASKGRAPH_UHR | 8.43 | 18346 |
| RSU_C_0339 | E | 200 | TASKGRAPH_UHR | 12.95 | 18240 |
| RSU_C_0358 | C | 200 | DIRECT_VLM | 13.44 | 23886 |
| RSU_C_0359 | A | 200 | DIRECT_VLM | 35.95 | 23890 |
| RSU_C_0366 | C | 200 | DIRECT_VLM | 9.82 | 23876 |
| RSU_C_0367 | E | 200 | TASKGRAPH_UHR | 6.38 | 18362 |

完整问题、选项、模型答案与错误字段见 [results.csv](uhr25_planner_20261011/results.csv)；逐条原始数据见 [results.jsonl](uhr25_planner_20261011/results.jsonl)。
这些逐题采集记录保存在本地及云端运行目录，未纳入 Git 提交。

该批次不改变服务的 120 秒协议预算。首题与之后的缓存热状态不同；不应将这一批的平均耗时推广为所有正式图像的耗时承诺。
