# Planner 程序化定位监督补充实施报告

基线为当前开发分支 codex/competition-protocol-v2，提交 12104c2；开始时工作树干净。本轮只实施最小 BBOX 类型合同补全、独立程序化数据工程及测试。没有训练、下载模型、修改权重、重建旧数据或修改既有训练配置；没有推送本轮改动。

## 结果与合同

已交付 2000 条英文、可直接由现有 Qwen3VLDataset 读取的 messages SFT：train 1800，validation 200，覆盖 40 个类别与当前六种固有属性字段。全部 FINAL 为 BBOX。单节点 LOCATE 1400 条（训练 1260，验证 140），占 70%。

| 族 | 样本数 | 比例 | 逻辑路径 |
| --- | ---: | ---: | --- |
| category | 900 | 45% | LOCATE→BBOX |
| intrinsic | 500 | 25% | LOCATE(固有属性)→BBOX |
| region | 300 | 15% | REGION→LOCATE→BBOX |
| extreme | 100 | 5% | LOCATE→SELECT_EXTREME→BBOX |
| ordinal | 100 | 5% | LOCATE→SELECT_ORD→BBOX |
| semantic | 100 | 5% | LOCATE→SELECT_SEM(SINGLE)→BBOX |

简单 bbox 的安全表达为 FINAL($n1,BBOX)。Planner 不输出数值框；FINAL 只引用一个有效视觉节点。LOCATE 的静态 EntitySet 类型保持不变，执行时必须验证恰好一个已决目标。不能用 CHOICE_SINGLE/MULTI 伪装无选项 bbox，也不能对多个目标求并集冒充单体。

~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("building",color="white",shape="square"))
FINAL($n1,BBOX)
~~~

~~~text
INTENT(OTHER)
n1=REGION($image0,BOTTOM_RIGHT)
n2=LOCATE($n1,T("warehouse"))
FINAL($n2,BBOX)
~~~

~~~text
INTENT(OTHER)
n1=LOCATE($image0,T("building"))
n2=SELECT_SEM($n1,null,"the building with trees immediately to its right",SINGLE)
FINAL($n2,BBOX)
~~~

比赛五类合同均完成 CPU/mock 检查：bbox 唯一视觉区域→原图 pixel_xyxy；integer 权威 ScalarInt→整数文本；single_choice 返回原始合法 ID；enum 返回合法枚举；short_text 返回长度和污染检查后的短文本。HTTP 仍恰为 protocol_version/request_id/item_id/answer。详细修改前及修改后源码核对见 protocol_contract_audit.md。以上均不是实模准确率证明。

## 数据如何产生及验证

结构化参数是唯一 Gold 源：family、category、intrinsic_attributes、region、extreme/ordinal/order/condition、template_id/template_family、split、seed、generator_version、intended_output_contract。同时渲染英文与构造最短 TaskGraph，不从问题正则反推标签。固有属性采用手工类别白名单组合；背景条件仅由 SELECT_SEM.query 消费，没有塞进 category 或增加背景 LOCATE。

每行输入由生产 CompetitionRequestAdapter.adapt 和 Qwen3VLPlannerProvider._messages 实际调用生成。原始问题单独记录 metadata.original_question；部署后缀记录 metadata.planner_question。user payload 为 question/question_type/choices/inputs，bbox 是 FREE_FORM、choices=null、image0 图像引用。符号图片路径不读取图像；封装尺寸和零 hash 只是构造接口的占位上下文，未声称已验证真实图像或目标。

messages 直接采用 system/user/assistant；assistant 仅规范 DSL。来源 synthetic_programmatic_plan，明确 synthetic_planning_assumption 与 visual_answer_verified=false；不伪造 answer_audit，也不改变原 PlannerSFTDataset accepted 门槛。metadata 记录生成配置、原问题、规则、版本、实际 schema/DSL/prompt/生产合同文件 SHA256 与逐项结构验证结果。现有版本字符串保留，增量扩展标识为 planner-bbox-final-v1。

2000/2000 均通过 Schema、Validator、TypeChecker、最短路径/结构化语义、生产输入精确比较、JSON→DSL→JSON 规范化无损往返以及约束完整接受。独立验证器检查每个样本的每字符前缀，共 157638 次；仅对完全相同字符串缓存结果，实际检查 19969 个唯一前缀。检查了配置配额、seed、类别归属、模板元数据与实际数据的一致性。

完整重新生成后逐文件 SHA256 相等，实际证明内容和顺序可复现，记录 reproducibility_verification.json。DSL 平均 77.819 字符，最大 168 字符。

真实本地 4B tokenizer（只加载 tokenizer，不加载模型）和正式 Dataset/PlannerTextDataCollator 对全部 2000 条做 CPU 检查：assistant-only loss mask 精确、无视觉张量、无意外截断。assistant 后缀平均 32.3515 tokens、最大 55，2000/2000 均低于 512 输出预算；完整序列平均 961.2935、最大 994，低于 1088。结果见 sft_encoding_verification.json。此检查耗时约 22.44 秒。随后补充显式 token 完整性检查并全量重验（24.78 秒）：所有 input 与有效 label IDs 均在 151669 词表范围，当前 tokenizer 没有定义 unk_token_id。记录在 sft_token_integrity_verification.json。

## 划分、旧数据与实际混合

训练/验证分别使用互斥的 30/10 个类别、10/3 个动作句式家族及独立上下文；区域/极值/序数顺序/外部语义条件的可分离取值也留出。精确文本重复、跨 split 原文、类别、动作家族和广义语言模板骨架交叉均为 0。旧 80 条英文问题只参与精确排重，交叉 0，没有将其当作新独立评估集。

同 split 内有意重复语言骨架，1935 行属于复用模板；没有宣称 2000 个语义完全独立任务。核心图骨架在两侧共享，单节点 LOCATE 本身没有可隔离的结构变化。验证集同时留出类别、句式和空间规则，属于组合分布外专项集，不能单维归因；vehicle/car 等父子类别也可能分属两侧。

独立混合工具已实际构建 planner_locate_mix_v1：保留全部 3934 条旧 train，从 1800 新 train 按族/类别无放回抽取 984，最终 train 4918；新增实际曝光 20.0081%。独立 synthetic validation 200，固定旧 test 217。混合后 FINAL：CHOICE_SINGLE 3853（78.3449%）、CHOICE_MULTI 81（1.6470%）、BBOX 984（20.0081%）。计数、关系、路径等原训练行全部保留，原训练配置未指向新目录。

固定 test 按原字节复制，SHA256 为 6e4b9b15c3809faa6b67b9264bbc44c7589e64e77de5eaabf5e3be22291f7571，与原文件一致。manifest 记录四个来源的 count/hash、抽样 ID、最终各类型与算子分布。

**实际发现的旧数据问题：**3934 条旧训练中有 118 条不能通过当前 DSL 类型检查（87 条 attribute_requires_singleton，31 条 select_result_not_visual_scope）；现有 train/test 原问题交叉 12 个。这些是历史数据与当前类型约束的兼容问题。本轮不修复或删除旧行，manifest 保留具体 ID 与失败代码；不能把整个混合集称为“当前 Validator 全通过”。新增 2000 条严格全通过，旧 accepted 导出门槛保持不变。正式再训练前应单独处理这些历史标签问题。

## 修改文件及原因

| 文件（仓库相对路径） | 原因 |
| --- | --- |
| src/sat_rs_vlm/taskgraph/schema.py | 生产 BBOX FINAL 类型及单一视觉 producer 准入 |
| src/sat_rs_vlm/taskgraph/runtime_types.py | unique_bbox_evidence，拒绝零/多/未决/不完整证据并保留 provenance |
| src/sat_rs_vlm/taskgraph/runtime.py | BBOX FINAL 在 defer/普通路径均强制唯一视觉证据，不额外 ChoiceResolver |
| src/sat_rs_vlm/competition/protocol_v2/answers.py | 复用唯一性保护，补原图尺寸一致性；保留旧 JSON 坐标兼容 |
| taskgraph_lab/taskgraph/enums.py | 实验室 BBOX 枚举 |
| taskgraph_lab/taskgraph/schema.py | BBOX FINAL 恰一 source |
| taskgraph_lab/taskgraph/type_checker.py | BBOX 专用 producer 和单体 SELECT 类型检查 |
| taskgraph_lab/taskgraph/validator.py | BBOX 无残余问题例外，避免把输出 bbox 误当输入 REGION_BBOX |
| taskgraph_lab/prompts/planner_student_system_prompt.txt | 简短 BBOX 定位规则，不改变 Planner 职责 |
| taskgraph_lab/prompts/system_prompt.txt | 教师合同同步，避免学生/教师 FINAL 不一致 |
| taskgraph_lab/specs/taskgraph_v1_1_dsl.md | BBOX DSL、基数及运行时失败契约 |
| taskgraph_lab/specs/taskgraph_v1_1_final_choice_contract.md | 明确 BBOX 与 choice FINAL 不同合同 |
| taskgraph_lab/configs/planner_synthetic_locate_v1.yaml | 可配置比例/数量/seed/预算 |
| taskgraph_lab/synthetic_locate.py | 结构化规则、真实输入复用、独立检查与统计 |
| taskgraph_lab/tools/generate_planner_synthetic_locate.py | 实际 SFT/清单/质量/人工审核导出，拒绝覆盖 |
| taskgraph_lab/tools/validate_planner_synthetic_locate.py | 独立重验、源记录/SFT一致性、每字符前缀与复现检查 |
| taskgraph_lab/tools/mix_planner_synthetic.py | 保护旧数据的独立、可配置混合构建 |
| taskgraph_lab/tools/verify_synthetic_planner_sft.py | 真 tokenizer 与正式 Dataset/collator 的 CPU 编码验证 |
| tests/unit/competition/test_bbox_planner_contract.py | 完整 bbox DSL→HTTP 及异常、五合同回归 |
| taskgraph_lab/tests/test_bbox_final.py | BBOX 类型/DSL/约束/旧 SELECT 与 FINAL 兼容 |
| taskgraph_lab/tests/test_synthetic_locate_generation.py | 生成路径、输入合同、去重、复现、防坏标签 |
| taskgraph_lab/tests/test_synthetic_planner_sft.py | Dataset/collator、mask、预算、混合及旧门槛保护 |

compiler.py/parser.py/constraint.py 已逐项检查，通用枚举与 Validator 路径自动支持 BBOX；无需修改其实现，新增测试证明往返与完整前缀可用。没有修改 LOCATE 路由、语义筛选算法/预算、CLIP、LEVIR-CC、竞赛字段或最终答案协议。没有新增依赖。

## 可复现命令

在仓库根目录 PowerShell 中使用指定 Python。以下工具拒绝覆盖已有生成目录/验证文件，应把输出改为新的目录或文件。

~~~powershell
$env:PYTHONPATH = 'src;.'
$py = 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe'
& $py -m taskgraph_lab.tools.generate_planner_synthetic_locate --config taskgraph_lab/configs/planner_synthetic_locate_v1.yaml --output-dir taskgraph_lab/data/planner_synthetic_locate_reproduce
& $py -m taskgraph_lab.tools.validate_planner_synthetic_locate --dataset-dir taskgraph_lab/data/planner_synthetic_locate_reproduce --every-prefix --output taskgraph_lab/data/planner_synthetic_locate_reproduce/static_verification.json
& $py -m taskgraph_lab.tools.verify_synthetic_planner_sft --input taskgraph_lab/data/planner_synthetic_locate_reproduce/train.jsonl --input taskgraph_lab/data/planner_synthetic_locate_reproduce/validation.jsonl --tokenizer 'E:\迅雷下载\Qwen3-VL 4B\Qwen3-VL-4B-Instruct' --output taskgraph_lab/data/planner_synthetic_locate_reproduce/sft_encoding_verification.json --batch-size 32 --max-seq-length 1088 --max-output-tokens 512
& $py -m taskgraph_lab.tools.mix_planner_synthetic --base-train taskgraph_lab/data/planner_sft_hard_curriculum_v1/train.jsonl --base-test taskgraph_lab/data/planner_sft_hard_curriculum_v1/test.jsonl --synthetic-train taskgraph_lab/data/planner_synthetic_locate_reproduce/train.jsonl --synthetic-validation taskgraph_lab/data/planner_synthetic_locate_reproduce/validation.jsonl --output-dir taskgraph_lab/data/planner_locate_mix_reproduce --synthetic-fraction 0.2 --seed 42
$testBase = Join-Path $env:TEMP ('pytest-locate-' + [guid]::NewGuid().ToString('N'))
& $py -m pytest -q --basetemp $testBase tests/unit/competition/test_bbox_planner_contract.py tests/unit/competition/test_protocol_v2.py tests/unit/competition/test_coordinates.py
& $py -m pytest -q --basetemp $testBase tests/unit/taskgraph/test_runtime.py tests/unit/taskgraph/test_counting_integration.py tests/unit/taskgraph/test_locate_select_v2.py tests/unit/taskgraph/test_final_choice_fusion.py
& $py -m pytest -q --basetemp $testBase taskgraph_lab/tests/test_bbox_final.py taskgraph_lab/tests/test_synthetic_locate_generation.py taskgraph_lab/tests/test_synthetic_planner_sft.py
& $py -m pytest -q --basetemp $testBase taskgraph_lab/tests/test_bbox_final.py taskgraph_lab/tests/test_final_contract.py taskgraph_lab/tests/test_dsl_roundtrip.py taskgraph_lab/tests/test_dsl_invalid.py taskgraph_lab/tests/test_dsl_integration.py taskgraph_lab/tests/test_dsl_constraint.py taskgraph_lab/tests/test_select_semantic_scope.py
~~~

数量/seed 可通过 --count/--seed 覆盖；--ratios 接受包含六个族权重的 JSON。超过独立模板容量直接报错，不重复样本凑量。混合占比超过本批无放回容量也直接拒绝。长命令遵守 AGENTS 后台日志/PID 规则；本轮实际运行的各批命令均在约 30 秒以内，无后台训练或推理任务。

## 测试及限制

生产合同/坐标组 141 passed、1 skipped；runtime/计数/LOCATE-SELECT/final fusion 79 passed；实验室相关回归 361 passed；新生成器与 SFT 工具 33 passed。合计 614 passed、1 skipped、0 failed；跳过是既有 Windows symlink 权限测试。最终检查汇总见 validation_summary.json；四组 XML 保存在本报告目录。首次工具自检中发现的旧语料 Parser 拒绝已通过显式兼容审计解决，没有放宽新数据验证。格式检查和 git diff --check 通过。

训练输入与部署消息一致，但本批没有真实图像/标注，所以目标存在性、唯一性、VLM/DINO 定位准确率、SELECT 条件判断、Planner 真正学会最短规划及训练后的旧能力保持都尚未验证。可训练数据与 mock 通过不是模型能力提升。本轮没有启动任何正式训练。
