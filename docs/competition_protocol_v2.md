# 正式比赛 Protocol 2.0 接入

本入口将密封图像包、比赛请求和受约束答案接到现有 TaskGraph 系统。
不改变 TaskGraph v1.1 schema、模型、LAE-DINO/COUNT 参数、检索算法或原 benchmark 默认路由。
没有新增依赖：使用已有 FastAPI/Uvicorn、Pydantic/Pillow，smoke 客户端使用 Python 标准库。

## 接入审计与架构

已有 runtime_from_config 真实 provider 工厂、整数/布尔/标签/全局 Region 类型、
InputComposer、ChoiceResolver、SemanticDecisionLayer 和双图处理。
原 HTTP /health、/infer 接受客户端文件路径，缺少 Bearer、密封包和比赛答案合同；
旧 ChoiceResolver 在各评分路径生成 A/B/C，不能原样接受比赛任意标识。

新增独立 competition/protocol_v2 包和专用 app。
运行时增加可选 execution_mode、defer_final_answer、choice_ids，默认值保持原行为。
比赛交接最终证据，禁用该请求的 final-only choice fusion，
不改逻辑 DAG、算子注册或 Route-4B 的职责。
benchmark adapter 在进入解析器前分配 A/B/C；
旧的直接 ChoiceRequest 调用保留兼容桥接，新比赛调用显式传 ID 和原始选项文本。

## 生命周期与显存

FastAPI lifespan 只初始化一次图像包目录和现有 runtime/provider 对象，退出时关闭。
生产配置拒绝缺省 fake、fixture planner 或 fake COUNT；
只有明确 offline: true 的独立配置允许离线测试替身，不能用 mock 掩盖未配置模型。

默认 server.preload: false：对象启动时初始化，惰性模型权重/侧进程在首次使用时加载，
该耗时包含在第一题的 120 秒中。preload: true 调用已有 provider 的 preload 钩子；
并非每个嵌套 LAE provider 都提供钩子，不能证明所有路径均已暖机。
显存紧张时保留惰性加载，在正式网站 smoke 前先执行本地真实配置 smoke，暖机需要使用的路径。
healthz 的 ready 表示服务及 manifest 已初始化，不代表真实模型准确率或全部暖机已验证。

只运行一个 Uvicorn worker。HTTP 无排队并发推理：忙时返回 503，推理与最终答案转换串行。
额外的 service 锁保护非 HTTP 调用；COUNT/检测内部沿用原 worker 和预取配置。
复用现有 2B choice provider；没有新增模型实例、GPU worker、batch 或预取 GPU 张量。
请求结束释放 composer 缓存及模型评分 session。

默认 120 秒超时返回 504。超时不强杀正在执行的 CUDA/侧进程；
资源释放前保持忙状态，避免下一题叠加显存。正常题目仍须在 120 秒内完成。

## 输入与图像包

从 manifest.json 加载 format=rsu-sealed-images-v1 和动态 dataset_id；
正式包替换只修改 COMPETITION_IMAGE_PACKAGE，无需修改代码。
目录为 assets/<32 位小写十六进制 asset_id>，不需要扩展名。
manifest 是图像目录，不能当作题目清单。

校验协议、1–2 张图、SHA-256/MIME 元数据以及有限答案集合。
首次使用流式计算 SHA-256，核对大小、实际 PNG/JPEG/WebP 格式和尺寸；
缓存依赖设备、inode、大小及纳秒修改/创建时间，文件变化后重新校验。
不在每题遍历并哈希整个包。路径限制在 assets 内，拒绝路径穿越和外部符号链接。
比赛包应只读挂载；哈希校验不是针对恶意本地并发替换文件的文件锁。
原文件不改名、不重新编码。现有模型/裁剪器按文件内容读图。

严格保留 images 顺序：第一张 before，第二张 after。
不按 asset ID、hash、文件名或 manifest 顺序排序。
hash/decode、路由、初始化、推理和序列化都包含在 HTTP 请求时间内。
请求不含 dataset/task_category/答案，本层不构造虚假的 XLRS/MME/VRSBench 元数据。
内部固定命名空间仅用于 trace，模式由比赛 adapter 显式指定。

供应的协议未完整枚举非 bbox 的 answer_type 字符串，因此不另造这些标签的白名单。
以 response_constraint.type 决定答案格式；显式校验 bbox 和
question_form=change_region 的 region_label 约定。

## AnswerContract 与统一 AnswerResolver

AnswerContract 只包含答案集合、选项文本、大小写、长度/整数下界和原图尺寸，
不含模型、算子、路由、GPU 或并行参数。

| type | 最终路径 | 返回与校验 |
|---|---|---|
| enum | 唯一匹配 typed value，否则 SemanticDecisionLayer → reason_and_decide 有限缓存评分 | values 原始拼写；case_insensitive 用于匹配；不从自由推理正则抽取 |
| single_choice | ChoiceResolver 显式传 values 和按 ID 对齐的 choices 文本 | 返回 ID，支持 opt-7、Z、B；非法/预计算非法 ID 报错 |
| integer | COUNT/整数几何的权威 ScalarInt | serializer 才转字符串；拒绝 bool、浮点、自由文本和低于 minimum 的值 |
| short_text | 优先 Label/LabelSet/Boolean/ScalarInt/Answer，否则既有 VLM 简短生成 | 不对已有结构化结果再问模型；校验 min_length、4096 字符 |
| bbox | singleton Entity/Region/EntitySet/RegionSet，或单框严格 JSON Answer | 显式坐标换算后返回原图像素数组；有限数字、正面积、图像归属/尺寸/边界正确 |

bbox 不接受自由推理文本、不挑多个目标中的第一个，也不用选项解析器。
支持明确声明的 normalized_0_1、scaled_0_1000/normalized_1000、percent_0_100 和像素坐标。
原图归一化乘实际宽高；crop-local/裁剪归一化需要明确原图裁剪框，先缩放再加 offset。
布局默认 xyxy，cxcywh 仅在来源明确声明时转换；拒绝越界/非数/模糊参考系，不静默 clip/clamp。
已有 global Region 默认保持像素，不重复执行检测/几何适配器已经完成的换算。
模型 Answer 仅接受完整的单框 JSON 数组或 {"label":...,"bbox":[...]}；
优先使用来源声明，无声明时可用 bbox.model_output_coordinate_format。
生产配置的 normalized_0_1 来自现有 grounding 训练提示，只作用于模型 JSON，不作用于 global Region。
数值大小不能区分小像素框与归一化框，不自动猜格式。
双图 bbox 由 Region 图像归属或模型明确 image_index 确定目标图，并核对请求尺寸。
转换的输入格式、参考系与输出像素坐标保存在内部 trace，不进入 HTTP 答案。

成功只返回 protocol_version、request_id、item_id、answer，回显不变，
HTTP 200 application/json。非 bbox 为字符串，总响应不超过 1 MiB。
内部 trace 留在 service 内存，不返回 trace、CoT、模型名、图像路径或候选列表。
短文本若包含明确的 think/analysis 标记或推理标题会拒绝输出，不用正则截取“最终答案”。

## 比赛路由

普通 enum、单选、短文本使用 DIRECT_VLM 交接图像证据，由统一 resolver 回答。
单图空间/关系/路径问题按语义进入 TASKGRAPH_UHR；大尺寸本身不触发 TaskGraph。
明确简单计数（当前直接匹配英语 How many…are there/visible 等）进入 DIRECT_DETECTION，
沿用真实 counting_system；不确定计数、关系计数/中文计数进入 planner + COUNT。
bbox 进入 planner/LOCATE/SELECT/全局几何路径。
双图普通变化题使用有 before/after 角色的原语义 provider；
双图整数/bbox 走结构化路径并保留时间顺序。
复杂 Route reasoning 继续使用原 Route-4B。

已分析开发包全部 125 张图片，分辨率与资源路由的建议见
[开发图片集分析与分辨率路由方案](competition_resolution_routing_plan.md)。该方案尚未实施。
2026-10-03 用户明确调整目标：入口只按分辨率和图像数量路由，题意由 planner 处理，
双时相统一使用现有低分辨率模型。以上题意/合同分支描述的是当前代码，后续实施将替换这些分支。
本地 LEVIR-CC SHA-256 验证确认开发集 50 张 256×256 图片字节完全匹配，组成 25 组源 A/B 时相对；
匹配结果用于来源报告，不加入线上路由。

### 运行记忆的联合哈希

运行记忆现在按原图文件字节的 SHA256 和原始问题 UTF-8 文本的 SHA256 精确匹配。
问题文本不去空格、不改大小写、不做 Unicode 归一化；哈希在补充 planner 指令前计算。
单图记录使用 image_sha256，多图记录使用按输入顺序排列的 image_sha256s；
图像数量和时相顺序都是键的一部分。比赛入口复用 ImagePackage 已实际校验的哈希，
常规运行时流式计算并按文件身份缓存 128 项，不解码图片，不增加显存使用。

只有图片和问题完整匹配且执行配置一致时，才复用旧表的 mode/variant。
缺少图片哈希、未知键、无效模式或同一键对应不同 mode/variant/backend 时均回退正常路由。
运行时不会只按 question_sha256 或 sample_id 路由；lookup(sample_id) 仅保留作旧表审阅接口。
每个请求重新应用裁剪配置，防止 tight 变体影响后续未命中请求。
该表仅保留既有实际运行配置，不导入答案或依据正确率选择配置。

scripts/taskgraph/prepare_hash_memory.py 提供 hash-images、verify-package 和 bind-table。
bind-table 校验问题哈希、图片路径覆盖和全部原字段，重放验证无冲突的联合键，
列出冲突并确认它们会回退，最后原子更新旧表。图片清单保存在
configs/taskgraph/mme_rs_image_sha256.jsonl；开发集和 MME 的来源匹配按文件字节比较。

## 错误与日志

协议/约束不合法 422，资产缺失 404，校验失败 422，鉴权失败 401，
推理/答案合同/bbox 失败 500，忙 503，超时 504。所有 HTTP 错误是固定简短 JSON。
不把异常转换成假的 Yes、0 或其他合法答案。
内部异常：ProtocolValidationError、UnsupportedConstraintError、ImageNotFoundError、
ImageHashMismatchError、AnswerContractViolation、BBoxValidationError、RuntimeInferenceError。

普通日志仅 request_id、item_id、约束类型、图像数、已解析尺寸、执行模式、耗时及固定 error code。
JSON 转义避免 ID 换行形成额外日志行。不打印 key、原始请求、完整题目、
图像字节或堆栈/本地路径。生产关闭 Uvicorn access log；
反向代理也不能记录 Authorization 或请求体。
bbox/logging 配置中的安全约定必须保持开启。

网页调试可使用独立的本机 offline/2B 入口，详见 [competition_local_test.md](competition_local_test.md)。
按需开启的私有 `logging.question_capture_path` 将有效请求题干、选项及图片描述符另存 JSONL，
默认关闭，不包含 API Key、请求头或图像字节；不要提交或公开该文件。

## 环境准备及真实部署

在仓库根目录，用现有模型环境运行：

~~~powershell
python -m pip install -e ".[model,retriever]"
$env:MODEL_API_KEY = (python -c "import secrets; print(secrets.token_urlsafe(32))")
$env:COMPETITION_IMAGE_PACKAGE = "<解压后的密封图片包绝对目录>"
$env:COMPETITION_CONFIG = "configs/competition/protocol_v2.yaml"
# 配置 runtime.real.example.yaml 中原有模型/adapter/LAE worker/GeoRSCLIP 环境变量。
python -m uvicorn sat_rs_vlm.competition.protocol_v2.server:create_app --factory --host 0.0.0.0 --port 8000 --workers 1 --no-access-log
~~~

runtime_config 引用原有模型工厂和参数，可以改为部署现有 runtime 的配置路径，
无需新模型配置格式。缺少 checkpoints 时默认配置不能直接运行真实服务。
LAE-DINO worker 环境和权重按原项目部署说明准备。
协议开发环境仅需 pip install -e ".[dev]"。Linux 用 export NAME=value 设置变量，服务命令相同。

AutoDL 等 Linux 云机可用仓库启动器运行完整 TaskGraph Protocol 2.0 服务：

~~~bash
cd /path/to/sat-rs-vlm
# /root/autodl_env.sh 可保存 MODEL_API_KEY 和自定义模型/LoRA 路径；启动器不会打印 key。
export QWEN3VL_4B_PLANNER_LORA="<本次选定的 Planner adapter 绝对目录>"
bash scripts/competition/start_cloud_protocol_v2.sh 8000
~~~

启动器默认使用云机已有的 2B/4B、ChangeChat LLM-only、GeoRSCLIP、LAE-DINO
及 `/root/autodl-fs/dev100-images`；环境变量可覆盖默认路径。它显式加入
`04_counting_system_plan/src` 并切换至部署仓库根目录，只启动一个 Uvicorn worker，
拒绝覆盖已有监听端口或运行目录。Planner adapter 必须显式指定，不默认冻结旧 checkpoint。
`AUTODL_ENV_FILE=/dev/null` 可跳过云机的通用环境文件。Uvicorn 配置会启用协议元数据日志，
记录实际路由、HTTP 错误与耗时，仍不记录题目全文或 API Key。
进程默认按需加载权重，`/healthz` 显示完整 runtime 是否装配成功及模型是否仍处于 lazy 状态；
收到真实推理请求前，需确认 GPU 已有足够显存。日志和 PID 写入独立的私有运行目录。

在配置好同一组云端路径后，无模型验收可运行：

~~~bash
CUDA_VISIBLE_DEVICES='' RUN_CLOUD_STARTUP_TESTS=1 "$PYTHON" -m pytest \
  tests/integration/test_competition_cloud_startup.py \
  tests/integration/test_competition_protocol_v2.py -q
~~~

第一组用正式生产配置检查真实组件装配，并禁止 Qwen/CLIP 权重加载及 LAE worker 启动；
第二组用明确的 fixture runtime 跑五种答案契约的本机 HTTP 请求。这些测试不证明模型答案准确率。

LAE-DINO 的隔离环境可能仍使用 Python 3.8。`instance_postprocess` 中立即执行的类型别名
必须使用 `typing` 的泛型，不能依赖 `from __future__ import annotations` 延迟别名求值。
设置 `LAE_DINO_PYTHON` 后，以下回归会在真实 worker 解释器中执行 `--help`，不加载检测器权重：

~~~bash
"$PYTHON" -m pytest tests/unit/integrations/test_instance_postprocess.py -q
~~~

验收真实模型服务时，除健康状态和响应格式外，还需检查实际路由、检测器 worker 日志，
以及 Planner/ChangeChat 共享管理器的 adapter 切换计数。合法的整数 `0` 本身不能证明
检测器执行成功；当前候选验证计数器在部分检测失败情况下会保留可输出零值的回退策略。
2026-10-13 云端完整链路验证与具体测试范围见
`reports/experiments/cloud_taskgraph_deployment_20261013/REPORT.md`。

长时间服务器按 AGENTS 规则后台启动，保存 PID、stdout/stderr；
Windows Start-Process 使用 -WindowStyle Hidden。以下 key 通过继承环境变量传递，不写入参数：

~~~powershell
New-Item -ItemType Directory -Force reports/competition | Out-Null
$competitionPython = (Get-Command python).Source
$competitionServer = Start-Process -FilePath $competitionPython -ArgumentList "-m uvicorn sat_rs_vlm.competition.protocol_v2.server:create_app --factory --host 0.0.0.0 --port 8000 --workers 1 --no-access-log" -WindowStyle Hidden -PassThru -RedirectStandardOutput reports/competition/server.stdout.log -RedirectStandardError reports/competition/server.stderr.log
$competitionServer.Id | Set-Content reports/competition/server.pid
# 状态：
Get-Process -Id (Get-Content reports/competition/server.pid)
# 停止（用户明确执行）：
Stop-Process -Id (Get-Content reports/competition/server.pid)
~~~

PowerShell 检查：

~~~powershell
Invoke-RestMethod "http://127.0.0.1:8000/healthz" -Headers @{Authorization="Bearer $env:MODEL_API_KEY"}
python -m sat_rs_vlm.competition.protocol_v2.smoke --endpoint http://127.0.0.1:8000 --package "$env:COMPETITION_IMAGE_PACKAGE"
~~~

smoke 支持 --api-key，优先环境变量避免参数历史暴露密钥。
五题只检查协议和合法性，不评估准确率；真实模型未满足合同会报错，不补合法假答案。
2026-10-03 按用户请求下载并验证开发图片包；没有注册网站、提交评测或消耗比赛次数。

## 本地合成 smoke

~~~powershell
python -m pip install -e ".[dev]"
python -m sat_rs_vlm.competition.protocol_v2.smoke --create-fixture .tmp/competition-images
$env:COMPETITION_IMAGE_PACKAGE = (Resolve-Path .tmp/competition-images).Path
$env:COMPETITION_CONFIG = "configs/competition/protocol_v2.offline.yaml"
# 同样设置 MODEL_API_KEY，然后使用上面的单 worker 启动命令。
python -m sat_rs_vlm.competition.protocol_v2.smoke --endpoint http://127.0.0.1:8000 --package "$env:COMPETITION_IMAGE_PACKAGE"
python -m pytest tests/unit/competition tests/unit/taskgraph -q
python -m pytest tests/integration/test_competition_protocol_v2.py -q
~~~

合成图两张 640×480，无扩展名；生成器拒绝覆盖已有 package。
offline 显式使用原 fake providers、fixture planner、真实 FastAPI/adapter/runtime 流程，
bbox fixture 用 REGION_FROM_BBOX 将 1000×1000 参考框换为原图像素。
这些是协议测试替身，不能用于比赛。
socket 集成测试创建短暂本地服务、跑真实 CLI 的五类请求，然后在测试范围内关闭。

## 官方检查和网站 smoke

官方 starter：https://gitee.com/masonJJJ/participant_api_starter。
取得 check_api.py 后，按所附 HTML 执行：

~~~bash
export MODEL_API_KEY='<与服务器一致的参赛 API 密钥>'
python3 check_api.py --origin https://your-public-domain.example --package /path/to/sealed-images
~~~

Windows 用 python check_api.py --origin <HTTPS origin> --package "$env:COMPETITION_IMAGE_PACKAGE"。
MODEL_API_KEY 是自设参赛接口密钥，不能用网站登录 token 替代。
生产 origin 是公网上有效证书的 HTTPS origin，不带路径、query 或末尾斜杠；
localhost、裸 IP、自签名、内网地址仅适合本地测试。

真实本地 smoke 和官方 check_api 通过后，在所附 HTML 指定网站手动注册 origin/API key，
确认预检 dataset_id 与加载包一致，再点击官方 8-question smoke。
通过后再申请 100-question full；得分性质按最新客观评测 PDF。
官方 smoke 每天 10 次、full 每天 2 次，UTC 重置；本工具不发起或停止这些评测。

## 验证范围和真实限制

覆盖 CPU/offline、TaskGraph、原 HTTP/CLI、配置/路由/类型测试及五类 socket smoke。
修复了两个已有测试替身问题：locator 测试用 -1 阈值绕过过滤，改为受支持的 None；
HF engine 测试补齐 telemetry 的离线 Transformers 子模块。
没有改定位阈值、模型效果或生产算法。

2026-10-02 使用的本地 .venv 无 torch，当次 KV-cache engine CPU 张量测试跳过；
2026-10-03 后测试使用用户指定的 WindowsApps Python 3.11.9，已有 torch，CPU 缓存评分测试可执行。
真实模型 smoke 需要 RUN_REAL_MODEL_TESTS=1、GPU 和本地 checkpoints。
Windows 未授权符号链接创建时该测试跳过；路径穿越/资产 ID 校验仍测试。
API/choice 非连续 ID、缓存调用已测试，真实 tokenizer 的多 token 候选评分仍需模型环境验证。
short_text 官方评分 normalization 未公开；简短生成不能保证得分。
复杂 bbox/关系计数取决于原 planner 和算子能否返回 singleton/global Region 或 ScalarInt；
不能确定时显式失败，不改成任意框/整数。
公网 TLS、平台 check_api、官方 8-question smoke 和 120 秒预算需在真实部署环境验证。

2026-10-02 本地最终验证：331 passed、0 failed、8 skipped，耗时 14.80 秒。
其中比赛测试及 socket smoke 52 passed/1 skipped，TaskGraph 239 passed，
核心/原 API 40 passed/1 skipped，真实模型 smoke 6 skipped。
Ruff（新增代码及修改的解析/运行时/评测文件）与 git diff --check 通过。
JUnit 在本地 .tmp/competition-protocol-v2-audit/results.xml，未加入仓库。
完整复现命令（PowerShell）：

~~~powershell
$competitionCoreTests = @(
  "tests/unit/test_hf_vlm_choice_cache.py",
  "tests/unit/test_hf_vlm_engine.py",
  "tests/unit/test_result_schema.py",
  "tests/unit/test_task_router.py",
  "tests/unit/test_task_protocol_and_metrics.py",
  "tests/unit/test_config.py",
  "tests/unit/test_model_factory.py",
  "tests/unit/test_environment_expansion.py",
  "tests/unit/test_vrsbench_adapter.py",
  "tests/integration/test_http_infer.py",
  "tests/integration/test_cli_infer.py"
)
python -m pytest tests/unit/competition tests/unit/taskgraph @competitionCoreTests tests/integration/test_competition_protocol_v2.py tests/smoke/test_taskgraph_real.py -q
~~~

2026-10-03 使用用户指定的 WindowsApps Python 3.11.9 运行上述回归范围：
372 passed、0 failed、7 skipped，耗时 25.16 秒。CPU KV-cache 张量测试已实际执行。
随后加强坐标来源声明一致性校验；最终比赛单元/HTTP/socket 测试 85 passed、1 skipped，耗时 4.51 秒。
跳过项为 Windows 符号链接权限及未显式开启的 6 项真实模型测试；没有运行模型准确率/性能评测。
真实开发包 125 张资产已通过现有 ImagePackage 校验，不代表真实推理或平台预检已完成。
测试解释器在本地 AGENTS.md 中记录，未将该机器路径加入共享配置。
本地结果文件为 .tmp/competition-protocol-v2-audit/results-20261003.xml 和
results-20261003-competition-final.xml，未加入仓库。
