# MME / XLRS 选择题 Planner 数据扩充报告

本轮在 `codex/competition-protocol-v2` 工作树中，将原始 2,815 条训练样本扩充为 **4,500 条**。新增 **1,685 条真实英文选择题**，使用 Codex 子 agent 审阅问题、编写有限转换规则，再由程序生成规范 DSL。没有调用 DeepSeek API、加载视觉模型或启动训练。此前生成的 BBOX 合成定位数据仍是独立数据版本，没有混入这 4,500 条。

## 交付结果

| 项目 | 数量或结果 |
|---|---:|
| 原训练样本 ID | 2,815 |
| 新增 MME_RealWorld_RS | 850 |
| 新增 XLRS_Bench | 835 |
| 最终训练样本 / 唯一样本 ID | 4,500 / 4,500 |
| 最终 MME / XLRS | 2,223 / 2,277 |
| 原标签在新版本中修订 | 176 |
| 原训练行按字节保留 | 2,639 |
| 固定测试集 | 217，按字节保留 |
| 新增题中含 SELECT_SEM | 694，约 41.2% |
| 新增 SELECT_SEM 节点 | 776：SINGLE 513 / MULTI 263 |
| 全训练集 SELECT_SEM 节点 | 989 |
| 最终 CHOICE_SINGLE / CHOICE_MULTI | 4,333 / 167 |
| 未完成、拒绝或缺失的新计划 | 0 |

主要文件位于 `taskgraph_lab/data/planner_choice_expanded_v1/`：

- `train.jsonl`：最终 4,500 条，训练输入是原问题、选项、符号图片引用；监督是 DSL。
- `new_train.jsonl`：仅新增的 1,685 条。
- `test.jsonl`：原固定 217 条的字节复制。
- `accepted_plans.jsonl`：本轮处理的 1,861 条，即 1,685 条新增加 176 条修订；不是完整训练集。
- `plan_audit.jsonl`：逐条 Validator、约束语法、往返、生产转换、来源答案格式检查及 review_notes。
- `dataset_manifest.json`：数量、来源、计划哈希、旧标签问题和发布文件 SHA256。

训练集 SHA256：`67d2fd13520b023e4159f08f2548d1df660c3d4b83dfea531487f6c49d88a55e`。
固定测试集 SHA256：`6e4b9b15c3809faa6b67b9264bbc44c7589e64e77de5eaabf5e3be22291f7571`。

## 原构建流程核对

原流程为 `build_seed_set.py` 从 MME / XLRS 源 JSON 正常化、分桶抽样，`generate_teacher_batches.py` / `generation/batch_generation.py` 调用 DeepSeek 转换，`revalidate_stage1.py` 进行图与来源答案检查，最后由 `prepare_planner_holdout_sft.py` 和严格的 `PlannerSFTDataset` 导出 DSL 消息。

原 2,815 条训练样本对应三批 accepted：971 + 1,730 + 114。`planner_sft_hard_curriculum_v1/train.jsonl` 的 3,934 是重复曝光后的课程采样行数，不能当成独立题量。本轮以 `planner_sft_hard_augmented_v1_20260901_173936/train.jsonl` 为基线。

原源题共约 6,818 条相关选择题：MME 遥感 3,738，XLRS 3,080。新抽样同时排除旧 train/test 的样本 ID 和规范化问题文本，并排除新增集合内部同文重复。抽样提示只用于检索和分配，不直接成为 gold 规划。新样本与旧测试集的 ID、规范化问题文本交集均为 0。

没有采用宽松的 JSON messages 导出通道；所有新计划必须经过真实 answer_audit、当前 Validator、DSL 往返和严格 SFT 导出。

## SELECT_SEM 与旧标签修订

新增题保留完整英文自然条件，使用 `SELECT_SEM(candidates,reference-or-null,query,SINGLE|MULTI[,scope])`。背景条件进入 `params.query`，只在已有候选内选择；类别与固有属性进入 LOCATE TargetSpec。明确的极值、面积排序、序数、分组和 COUNT 保留专用算子。

二次审阅修正了参考物位置被误用为目标整图裁剪、先排序再筛范围、名词头误识别、屋顶颜色属性归属，以及复数集合直接交给单体 ATTRIBUTE 等问题。程序输出逐题 `rule_id`、`review_notes`；有限审阅转换规则有源内容或问题指纹保护，不能直接把旧规则用于未来任意抽样。

旧标签仅在新版本修订，原始 train/test 与源 JSON 哈希均未改变：

1. 103 条当前类型合同问题，主要为集合 SELECT_REL 后直接 ATTRIBUTE，以及把非视觉 scope 的 SELECT 结果交给 LOCATE / COUNT_IMAGE。
2. 45 条生产排序合同问题：实验室允许的 `height`、`width`、`length`、`distance` 等不能被生产端执行。改用完整条件的语义筛选，合法面积排序、分组、路线与计数保留；没有把长度、宽度、高度或距离偷换成 bbox_area。
3. 34 条旧 HRSCD 双时相输入问题，其中 6 条与第 1 项重合，所以总修订数为 103 + 45 + 34 − 6 = **176**。源数据用冒号连接两个 HRSCD 图片键；新版本显式绑定 image0/image1，分别在红圈指定区域计数，再 ABS_DIFF，不假设左右拼接。新增题中同类问题另有 2 条，也已规范化。原始复合键保留在审计 metadata。

修订行使用支持 SELECT_SEM 的当前系统提示词，避免旧提示词禁止该算子却监督模型输出它。未修订旧行保留原字节；修订行的问题和选项不变，只有上述双图输入绑定按源事实展开。

## 实际检查

最终 **606 项测试通过，0 失败**，包括完整 Planner 实验室测试以及生产 Planner 桥、LOCATE / SELECT 和路由回归。新增测试覆盖缺失/重复/越界计划、问题泄漏、答案字段泄漏、来源选项与 cardinality、固定测试保护、双图绑定、非法 bbox、单体/集合颜色、旧排序生产拒绝和修订后放行。本轮新增代码 ruff 通过；有限审阅描述表的长行及直接脚本路径初始化有明确的风格例外。

全部 4,500 条训练标签又通过真实 `Qwen3VLPlannerProvider._to_production_graph` 转换检查，不将实验室 Schema 合法等同于生产可执行。7 个转换模块重新生成后，所有计划文件哈希与发布清单一致。原始训练集、固定测试集、MME/XLRS 源 JSON 哈希复核均不变。

第一次生产回归使用已有 `.tmp/pytest` 时遇到 Windows 临时目录权限错误；改用新的独立临时目录后通过，最终 606 项结果使用独立临时目录。没有修改权限或清理用户文件来绕过问题。

真实本地 4B tokenizer 对全部 4,500 条进行了无截断编码、prompt/assistant 前缀与 token ID 完整性检查；对每个 1,500 条分段中最长的 16 条，共 **48 条**，实际调用正式训练 collator 验证 assistant-only loss mask 和无视觉张量。

| 编码指标 | 结果 |
|---|---:|
| 最长完整训练序列 | 1,472 tokens |
| 最长 assistant DSL 后缀 | 258 tokens |
| 超过旧 max_seq_length=1088 | 275 条 |
| 超过输出预算 512 | 0 条 |
| 后续训练建议 | max_seq_length=1536，max_new_tokens=512 |

这些是 tokenizer 和静态检查，不是模型推理实验。没有修改现有训练配置或自动启动训练；使用新数据前应调整训练序列长度，不能沿用 1,088 并静默截断。

## 成本与限制

本轮 DeepSeek API 调用为 **0**，使用了当前 Codex 账户用量；中途确实因用量限制暂停，恢复后续做。因此没有新增 DeepSeek API 账单，不代表没有计算用量成本。按旧两批调用记录外推，若改用原 API 流程，约需 445–455 次批量/修复调用、输入约 360–368 万 tokens、输出约 200–214 万 tokens；这只是旧提示词与旧接受率下的估算，不能当成当前报价或本轮实耗。当前停在约定的 4,500 条，没有继续扩大。

没有新增依赖、下载参考图片、修改推理路由/预算、模型权重、竞赛接口或最终答案协议。本轮只增加数据准备、有限转换、审计、SFT 导出和测试工具，工作树原有修改保留。

没有进行原图 VLM/DINO 推理、人工框标注验证、4B Planner 重训或新数据准确率对比。来源 answer_audit 只检查选项标签、格式与单/多选合同，不能证明生成的规划在图像上一定正确。区域候选、最高/最长判断、复杂原因及翻译歧义仍需真实模型实验；部分旧路线的范围外规划问题也没有被本轮全面重标。

固定旧测试集保持字节不变，但其中 **14 条**标签不能通过当前生产合同，详情见 `release_verification.json`；旧 split 还有 12 个同文问题交集，未据此认定为相同图像样本。不能将这些遗留情况计为新样本准确率退化，也不能拿静态通过证明准确率提升。后续应另建经过审核的 SELECT_SEM 评估集，并单独修订/版本化旧测试标签。

## 本轮新增文件及用途

| 文件 | 用途 |
|---|---|
| tools/prepare_choice_expansion.py | 源题去重、抽样、分片与来源哈希清单 |
| tools/choice_source_inputs.py | 仅展开明确 HRSCD 双图键，不猜测一般冒号/盘符/URL |
| tools/codex_choice_bbox.py | 三种已审阅的显式参考框模板转换，共 386 条 |
| tools/codex_choice_mme_a.py / codex_choice_mme_b.py | 两组各 425 条 MME 审阅规则和 DSL 生成 |
| tools/codex_choice_xlrs.py | 449 条非显式框 XLRS 审阅规则和 DSL 生成 |
| tools/codex_choice_legacy.py | 原类型与双图绑定修订，共 131 条 |
| tools/codex_choice_rank_mme.py / codex_choice_rank_xlrs.py | 18 + 27 条旧排序生产合同修订 |
| tools/collect_choice_expansion.py | 严格聚合、真实生产桥门禁、来源答案审计和独立版本导出 |
| tools/verify_choice_sft.py | 本地 tokenizer 分段核查及最长样本正式 collator 检查 |
| tests/test_choice_expansion.py / test_choice_plan_generators.py | 聚合器、泄漏保护、输入规范化与有限转换边界测试 |
| data/choice_expansion_codex_v1/ | 无答案种子、审阅分片、逐条计划与分片审核记录 |
| data/planner_choice_expanded_v1/ | 训练、增量、固定测试、审计与最终发布清单 |
| reports/taskgraph/planner_choice_expanded_v1/ | 本中文报告、生产复核、编码、测试和可复现证据 |

表中 tools/tests/data 路径均位于 taskgraph_lab。其他工作树中的 BBOX 生产代码修改属于之前任务，本轮没有覆盖。

## 复核命令

所有 Python 命令使用指定解释器。下列短时检查均已执行，没有启动长期后台任务。

~~~powershell
$py = 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe'
$env:PYTHONPATH = 'src;.'
$testBase = Join-Path $env:TEMP ('pytest-choice-check-' + [guid]::NewGuid().ToString('N'))
& $py -m pytest -q --basetemp $testBase taskgraph_lab/tests tests/unit/taskgraph/test_planner_provider.py tests/unit/taskgraph/test_locate_select_v2.py tests/unit/taskgraph/test_locate_routing_v2.py
& $py -m ruff check taskgraph_lab/tools/codex_choice*.py taskgraph_lab/tools/choice_source_inputs.py taskgraph_lab/tools/prepare_choice_expansion.py taskgraph_lab/tools/collect_choice_expansion.py taskgraph_lab/tools/verify_choice_sft.py taskgraph_lab/tests/test_choice_expansion.py taskgraph_lab/tests/test_choice_plan_generators.py
& $py -m taskgraph_lab.tools.collect_choice_expansion collect --prepared-dir taskgraph_lab/data/choice_expansion_codex_v1 --output-dir taskgraph_lab/data/choice_expansion_recheck --base-train taskgraph_lab/data/planner_sft_hard_augmented_v1_20260901_173936/train.jsonl --fixed-test taskgraph_lab/data/planner_sft_hard_augmented_v1_20260901_173936/test.jsonl --mme-json D:/Desktop/tzb-2026/MME_RealWorld.json --xlrs-json D:/Desktop/tzb-2026/xlrs_questions_answers.json --target-count 4500 --validate-only
~~~

重新导出时使用新的空输出目录并移除 `--validate-only`；工具拒绝覆盖已有正式 train.jsonl。编码核查使用 `verify_choice_sft`，分别指定 `--offset 0/1500/3000 --limit 1500`，tokenizer 路径和准确训练哈希保存在 `encoding_summary.json`。

## 真实源题的规划标签示例

下列是从最终新训练样本抽取的生成标签，均已通过生产转换。它们不是模型预测结果，也不包含来源 GT 或视觉答案。

### 1. MME：单体语义筛选后读取属性

样本：`mme_rs_perception_remote_sensing_color_2922`。

原问题：What color is the roof with a special color in the top right corner of the picture?

~~~text
n1=REGION($image0,TOP_RIGHT)
n2=LOCATE($n1,T("roof"))
n3=SELECT_SEM($n2,null,"the roof with a special color in the top right corner of the picture",SINGLE,$n1)
n4=ATTRIBUTE($n3,"color")
FINAL($n4,CHOICE_SINGLE)
~~~

### 2. MME：候选集合语义筛选后计数

样本：`mme_rs_perception_remote_sensing_count_2815`。

原问题：How many cars are parking beside the circular greenland in the upper left area of this picture?

~~~text
INTENT(RELATIONAL_COUNT)
n1=LOCATE($image0,T("car"))
n2=SELECT_SEM($n1,null,"cars parking beside the circular greenland in the upper left area of this picture",MULTI,$image0)
n3=COUNT_ENTITIES($n2,T("car"),false)
FINAL($n3,CHOICE_SINGLE)
~~~

### 3. XLRS：语义定位与专用空间比较

样本：`xlrs_000612`。

原问题：In the image, what is the orientation of the gray square building with a pointed roof in the town in the lower right corner relative to the white building with a pointed roof next to the trees?

~~~text
n1=LOCATE($image0,T("building",color="gray",has_part="pointed roof",shape="square"))
n2=SELECT_SEM($n1,null,"the gray square building with a pointed roof in the town in the lower right corner",SINGLE,$image0)
n3=LOCATE($image0,T("building",color="white",has_part="pointed roof"))
n4=SELECT_SEM($n3,null,"the white building with a pointed roof next to the trees",SINGLE,$image0)
n5=RELATION($n2,$n4)
FINAL($n5,CHOICE_SINGLE)
~~~

### 4. XLRS：给定参考框的属性任务

样本：`xlrs_001590`。

原问题：Determine the color of the object within the given reference bounding box in the satellite image. Image resolution: 10000 x 10000. Bounding box: [4106, 3314, 4324, 3606].

~~~text
INTENT(ATTRIBUTE_QUERY)
n1=REGION_BBOX($image0,[4106.0,3314.0,4324.0,3606.0],[10000,10000])
n2=ATTRIBUTE($n1,"color")
FINAL($n2,CHOICE_SINGLE)
~~~

### 5. XLRS：独立双图的计数差

样本：`xlrs_001929`。

原问题：In the areas marked by red circles in the images, how many differences are there between the cars in the two pictures?

~~~text
n1=FIND_MARKER($image0,"circle","red")
n2=COUNT($n1,T("car"),true)
n3=FIND_MARKER($image1,"circle","red")
n4=COUNT($n3,T("car"),true)
n5=ABS_DIFF($n2,$n4)
FINAL($n5,CHOICE_SINGLE)
~~~
