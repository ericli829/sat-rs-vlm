# Protocol 2.0 本机服务与网页调试

`local_server:create_local_app` 是独立调试入口，不执行正式 planner、检测器、检索器或哈希记忆，
也不改动正式入口的路由。它读取真实开发图片包，并核验 asset ID、SHA256、MIME 和尺寸。

- `offline`：所有五种答案格式返回明确的测试占位值，只验证协议。
- `qwen2b`：整图直接调用一份 Qwen3-VL-2B 与指定 LoRA。枚举、单选复用已有 KV 缓存评分；
  文本、整数和 bbox 直接生成，输出不满足约束时返回错误，不以占位值补齐。
  这个 LoRA 用于 LEVIR-CC 训练，不保证计数、定位或其他数据集的准确率。

2B 默认使用 bfloat16、262144 max_pixels、96 个新 token、一名 worker、串行推理，
不加载 4B 或检测器。模型在接受请求之前加载。这里的 bbox 提示明确要求 0–1 归一化坐标，
通过已有转换器转为原图像素，并校验边界；不凭数字大小猜测坐标格式。

PowerShell 启动本地服务：

```powershell
& .\scripts\competition\start_local_protocol_v2.ps1 -Mode qwen2b
# 或 -Mode offline
# Base model only (no LoRA is loaded):
& .\scripts\competition\start_local_protocol_v2.ps1 -Mode qwen2b -NoAdapter
```

脚本按仓库规则后台运行，返回 PID 和文件路径，不等待模型加载完成。
默认包为 `D:\Desktop\tzb-2026\dev100-images`，基座为同级 `Qwen3-VL-2B-Instruct`，
LoRA 为 `results\levir_train\vrsbench_levircc_replay_formal\round_2_adapter\checkpoint-3406`。
可以通过参数覆盖，不修改 checkpoint 的 adapter_config.json。
`-NoAdapter` 将配置中的 `model.adapter_path` 明确设为 `null`，使用已有加载器的纯基座路径。
启动器不会自动停止现有服务；显存不足以同时容纳两份模型时，应先明确安排服务切换。

运行目录在 Git 忽略的 `.tmp` 中：

- `api-key.txt`：随机模型 API Key；网站提交时使用，不能用队伍登录令牌替代。
- `questions.jsonl`：本地私有题目记录，每个经过鉴权、协议解析的请求写一行。
  包含原题干、选项、答案约束、按原顺序的图片描述符、请求 ID、UTC 时间和题干 SHA256；
  不含请求头、API Key、图像字节。超时或服务忙的有效请求也会在推理前记录。
- `stdout.log`、`stderr.log`：启动与运行日志，普通日志仍不写题目全文。
- `state.json`、`server.pid`：进程与配置路径。

正式工厂的题目记录默认关闭；可明确设置 `logging.question_capture_path` 启用独立私有文件，
普通日志 `request_metadata_only` 仍保持 true。不要将这个私有文件提交到仓库或公开。

比赛网页需要公网有效证书的 HTTPS origin，本地 `127.0.0.1` 不能直接填写。
`prepare_public_protocol_v2.py` 供后台运行：它启动本地服务，下载 Cloudflare 官方
`cloudflared` Windows 工具并校验固定版本 SHA256，再建立临时 Quick Tunnel。
工具保存在 `.tmp/tools`，不新增 Python 依赖。
下载中断时保存已有字节，以 HTTP Range 续传并自动重试；若下载服务器忽略 Range，
则从头覆盖下载。只有完整文件通过固定大小和 SHA256 检查后才执行工具。

网页准备任务另外写：

- `preparation.status.json`：阶段、子进程 PID、下载进度和公网检查结果。
- `endpoint.txt`：生成的 HTTPS origin；不要加接口路径或末尾斜杠。
- `tunnel.stdout.log`、`tunnel.stderr.log`、`tunnel.state.json`：隧道日志与进程。
- `job.json`：后台准备任务 PID 与开始时间。

如果本地服务已经启动而隧道准备中断，可在后台给准备脚本加 `--reuse-server`，
复用原运行目录及 API Key，只建立隧道。恢复任务身份保存在 `recovery.job.json`。
启动器输出重定向到文件，避免后台服务继承管道导致准备任务等待日志 EOF。

只有 `preparation.status.json` 中 `stage=ready` 才表示公网 `/healthz` 已校验成功，
开发包编号正确，并且无 Bearer 请求返回 401；这不表示五种题型都经过真实模型验证。
任务不会登录比赛网站或消耗 smoke/完整开发集次数，网页评测由用户发起。
保持电脑、服务、隧道在线；临时域名在隧道停止后失效。

查看状态和停止本次服务（替换运行目录）：

```powershell
& .\scripts\competition\get_local_protocol_status.ps1 -RunDirectory '<运行目录>'
& .\scripts\competition\stop_local_protocol_v2.ps1 -RunDirectory '<运行目录>'
```

只查看或复制 API Key：`Get-Content '<运行目录>\api-key.txt'`。
收到网站问题之后，可从 `questions.jsonl` 分析题型、图像及问题哈希。
# Change-region prompt replay

`scripts/competition/replay_change_region.py` can replay captured `change_region`
requests against an already running local diagnostic service. It loads no second
model and sends requests serially. Each pair receives an original enum baseline,
a generated change caption, and a generated region label guided by that caption.
The latter two calls use `short_text`, so the last answer is generated directly
instead of selected by finite-label likelihood scoring.

The experimental prompt assumes equal 3x3 cells and uses the center of the
dominant structural change cluster. These are experimental conventions, not
definitions verified against the official scorer. Images keep their original
before/after order. Full responses and timing are saved privately in the run
directory; keys are not written into results. Invalid labels remain invalid.

`report.html` embeds the original images and offers manual region annotation,
comparison of human accuracy, and JSON export. Original website correctness
flags do not reveal gold labels and are not used to guide either model pass.
Manual accuracy is not official accuracy. Replayed baseline labels cannot be
asserted to be identical to the earlier website predictions, which were not saved.

Use `start_change_region_replay.ps1` with `-StateFile`, `-SourceRun`, `-ScoreFile`,
and a fresh `-RunDirectory` to launch in the background. Inspect `status.json` or
use `stop_change_region_replay.ps1 -RunDirectory ...` to stop only the replay
client. The existing model service is not restarted or stopped.

For repeated trials, `repeat_change_region.py` generates a fresh caption per
item and trial, then sends the same caption and region prompt through free
generation and enum selection. It reuses the saved original baseline and a
user-exported human review file for local accuracy calculation. Human labels
never enter model requests. Add `-RepeatRounds 3 -HumanReview ...
-BaselineResults ...` to the background launcher for this mode. The report
and summary show each trial's strict accuracy, direction-alias-only accuracy,
legal-output counts, and per-item label variation. Unclear human labels are
excluded, and invalid answers count as incorrect.

This measures the existing decoding paths: free generation inherits the local
base generation config's sampling settings, while enum cached reasoning
explicitly uses `do_sample=False`. It is not a sampling-controlled comparison
of prompts alone. Alias normalization is an analysis metric and does not change
the service response or introduce an answer fallback.
