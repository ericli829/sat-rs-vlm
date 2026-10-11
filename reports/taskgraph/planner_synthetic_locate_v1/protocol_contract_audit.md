# Planner 定位补充工程：修改前合同审计

审计基线：当前开发分支 codex/competition-protocol-v2，提交 12104c2。审计开始时工作树干净。本文先于本轮代码修改和数据生成落盘。下列源码位置以该提交为准。

## 正式输入链路

models.py 的 ResponseConstraint（26–54）支持 bbox、integer、single_choice、enum、short_text。bbox 必须声明 pixel_xyxy、length=4，PredictRequest（72–102）要求 bbox answer_type 一致并声明原图尺寸。single_choice 的 choices 键须恰好覆盖 values。最终序列化由 AnswerContract.from_request（119–139）从 response_constraint 决定，不能把 Planner 的内部 answer_type 当成 HTTP 合同。

CompetitionRequestAdapter.adapt（adapter.py:37–105）把 bbox 强制路由到 TASKGRAPH_UHR/localization，并在原题后实际附加：

> Resolve the referred target using localization and selection. Final evidence must be one Entity or Region in original global pixel coordinates.

bbox 使用 FREE_FORM、空 options，且 defer_final_answer=True。Runtime 在 runtime.py:490–496 构造 PlannerRequest：question、question_type、options、ImageRef 映射、sample_id。Qwen3VLPlannerProvider._messages（providers.py:627–658）仅向 Planner 提供 question、question_type、choices（没有选项时为 null）和 inputs。inputs 的 key 是 image0，值为 type=image、uri_or_key。DSL 引用使用 $image0。HTTP response_constraint、原图尺寸、sha256 不直接进入 Planner user payload。

新增数据必须调用该 Adapter 和 _messages 复用实际行为；原始题与附加指令后的 question 分开记录。符号图像绑定不等于真实图像存在或经过答案审核。

## 五种输出合同

| wire 合同 | 答案层所需最终证据 | 与内部 FINAL 的关系 | 当前证据 |
| --- | --- | --- | --- |
| bbox | 单个 Entity/Region、恰好一个元素的 EntitySet/RegionSet，或明确坐标空间的严格 JSON Answer | 内部没有 BBOX；现有夹具用 TEXT+Region，存在类型合同缺口 | answers.py:149–204；competition/planner.json:12 |
| integer | ScalarInt，exact int，满足 minimum；不能用自由文本近似 | INTEGER 对应结构化 COUNT/整数几何证据 | answers.py:140–148,224–228 |
| single_choice | ChoiceResolver 按原始 values/option_texts 判定，返回真实合法 ID | CHOICE_SINGLE 保留选择题语义 | answers.py:104–116 |
| enum | 可直接规范匹配的 typed 文本，或有限候选语义决策 | LABEL/BOOLEAN/TEXT 等证据可进入答案层；不是一对一 FINAL 名称映射 | answers.py:117–139 |
| short_text | typed 文本直接使用，否则进行短文本语义生成；严格检查长度及污染 | TEXT/LABEL 等结构化结果可使用 | answers.py:205–209,235–244 |

以上五类已有 CPU/mock 测试源代码覆盖；这不是本轮已运行证明，更不是模型准确率证明。现有 test_http_auth_echo_five_contracts_and_safe_logs 验证 HTTP 五类响应，test_parse_contracts 验证请求。integer 的结构化无重问测试为 test_graph_integer_returns_scalar_without_reasking。

## defer_final_answer 和 HTTP

executor.py:307–314 在 defer 时关闭最终 choice fusion；runtime.py:574–577 关闭 choice_requested，633–642 返回原结构化 source；DIRECT_VLM 的延迟路径位于 703–705。bbox 答案分支不调用 ChoiceResolver。serialize_response（answers.py:245–250）仅输出 protocol_version、request_id、item_id、answer 四字段。正式 HTTP 字段不需修改。

## BBOX 内部合同缺口及最小修复边界

- 生产 schema.py:45–52 和实验室 enums.py:35 均缺 BBOX。实验室 FinalSpec 要求非空 question，而 validator.py:511 无条件要求视觉证据残余问题，导致无法表达无需额外 VLM 生成的 FINAL($n1,BBOX)。
- type_checker.py:99 的 LOCATE 静态输出为 ENTITY_SET；最终检查 241–259 仅采用联合类型相交。BBOX 需要单来源、明确单体需求的专用准入规则，不能将 SEMANTIC MULTI、RELATION 或 GROUP 集合作为唯一框。
- 单节点 LOCATE 必须保留。其单体物理路由已由 locate_routing.py:104–108 的 direct_localization 规则决定，不需改变路由规则。BBOX FINAL 在运行时必须验证恰好一个有效目标；静态准入不声称图像中存在唯一目标。
- compiler.py:253–267、parser.py:350–361 和 constraint.py:62,300–307 使用通用 AnswerType 枚举表面；扩展枚举后仍需测试解析、编译、前缀及最终基数限制。
- validator.py:363 目前把输出 bbox 词误认为必须输入 REGION_FROM_BBOX，应对 BBOX FINAL 排除该误警告。
- answers.py 现有 0/多目标集合拒绝、坐标有限/正序/边界、归属原图路径检查已存在，但 source.image 自带尺寸与原图尺寸不一致尚未统一拒绝；嵌套未决/不完整证据也需在 bbox 答案层统一保护。

新增 CPU/mock 测试应覆盖完整请求→生产 Planner 消息→DSL→TaskGraph→单目标证据→AnswerResolver→四字段 HTTP 响应，以及局部裁剪坐标、0/多、未决、NaN/越界/反序、图像归属/尺寸和禁止 ChoiceResolver。既有五类 wire 合同必须回归。

## 现有训练覆盖与导出边界

当前 4B hard_refine 配置使用 planner_sft_hard_curriculum_v1/train.jsonl（3934）和固定 test.jsonl（217），max_seq_length=1088、max_new_tokens=512。训练样本中：包含 LOCATE 2488，SELECT 1796，COUNT 1560；只有一个 LOCATE 节点的样本 163。FINAL 为 CHOICE_SINGLE 3853、CHOICE_MULTI 81；FREE_FORM 和 BBOX 都为 0。因此仅增加更多英文问题、继续用 CHOICE_SINGLE 标签无法弥补 bbox 合同监督。

PlannerSFTDataset（planner_dataset.py:99–112）对 accepted 记录同时要求 runtime_validation.valid 和 answer_audit.valid；程序化无图规划不能伪造视觉答案审核。Qwen3VLDataset（qwen3vl_dataset.py:17–24,90–99）支持直接 messages JSONL，可采用独立严格验证的 synthetic_programmatic_plan 通道，保留旧 accepted 门槛。

PlannerTextDataCollator 复用正式 assistant loss mask，并对纯文本 Planner 输入跳过视觉编码。新增数据不需要模型权重。原始训练、hard curriculum、固定 held-out test 和训练配置均保持原位；混合构建应只写新目录，并报告实际新增曝光比例。

## 证据限制

本次审计没有运行真实视觉模型、没有训练、没有验证真实目标存在性。合成标签必须明确记录 synthetic planning assumption。正式数据生成和修复后测试的实际结果另见本目录后续实施报告及数据目录质量报告。

## 补充：其他可访问训练导出覆盖（修改前统计）

以下 LOCATE/SELECT/COUNT 是包含该算子的样本数，不是节点总数。一节点 LOCATE 指图仅有 LOCATE 节点。

| 数据集 | 行数 | LOCATE | SELECT | COUNT | 一节点 LOCATE | BBOX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hard_curriculum train | 3934 | 2488 | 1796 | 1560 | 163 | 0 |
| hard_curriculum test | 217 | 128 | 97 | 98 | 4 | 0 |
| hard_augmented train | 2815 | 1388 | 903 | 985 | 163 | 0 |
| new2701 train | 2701 | 1278 | 827 | 970 | 163 | 0 |
| planner_sft_v1 train | 193 | 113 | 84 | 88 | 4 | 0 |
| planner_sft_v1 validation | 24 | 15 | 13 | 10 | 0 | 0 |

当前 hard_curriculum 的 SELECT 节点模式：ORDINAL 54、RELATION 1992、RANK 453、EXTREME 102、SUBREGION 162、SEMANTIC 0。3934 曝光行来自 2815 个唯一源样本。固定测试集为 217 条；现有训练/测试已有 12 个规范化原问题交叉，未在本轮改动。

## 修改后核对与实际测试

生产 schema.py:53 与实验室 enums.py 均引入 BBOX。生产 schema.py:530 和实验室 type_checker.py:154,269 限制 FINAL 单一视觉来源：LOCATE、REGION/REGION_FROM_BBOX 或单体 SELECT（RANK/ORDINAL/EXTREME/SUBREGION/SEMANTIC SINGLE）。排除 RELATION、MULTI、GROUP、文本与标量。

DSL 为 FINAL($n1,BBOX)，可无残余问题。生产转换会为现有 FinalSpec 填入空问题，运行时不为 BBOX 再次提问。compiler/parser/constraint 使用通用 AnswerType 枚举和既有 Validator，新增类型自动进入表面语言；无需为 BBOX 添加特殊解析器分支。已有四/五参数 SELECT_SEM 与其他 FINAL 模式保持回归兼容。

runtime_types.py:234 的 unique_bbox_evidence 在 runtime.py:626 以及 answers.py:162 强制恰好一个已决 Entity/Region，并检查嵌套包装的未决、不完整状态；不合并多框，不制造全区域成功目标。answers.py 同步验证源图像声明尺寸与封装原图一致。LOCATE 后端路由、检测/VLM 预算及接口字段没有改动。

CPU/mock 完整链路已运行：比赛 bbox 请求→生产 Adapter→真实 Planner 消息格式→mock DSL→TaskGraph 执行→唯一 Entity/Region→AnswerResolver→原图 pixel_xyxy→四字段 HTTP。覆盖局部 crop 缩放回原图、零/多目标、未决/不完整、非法/NaN/反序/越界坐标、图像归属/尺寸以及禁止 bbox 调用 ChoiceResolver。

生产合同与坐标组实际 141 passed、1 skipped（既有 Windows symlink 权限）；运行时/计数/LOCATE-SELECT/final choice 回归 79 passed。实验室 BBOX/DSL/SELECT scope/FINAL 回归 361 passed。测试名称、命令及 XML 见 implementation_report.md。五种比赛答案均已验证到 CPU/mock 格式与数据流层；本轮没有任何真实模型能力验证。enum/short_text 的语义 VLM 以及 single_choice 的候选判定准确率仍需实模实验。
