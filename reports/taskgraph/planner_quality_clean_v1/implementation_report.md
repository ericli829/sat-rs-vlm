# Planner 训练数据清洗报告

本轮实际清洗了旧 MME/XLRS 标签、最新扩充选择题、合成定位题及旧测试题。
原数据文件保持不变，结果独立保存在 `taskgraph_lab/data/planner_quality_clean_v1/`。
没有训练模型、调用付费 API、修改权重或修改推理执行器。

## 数据结果

| 数据 | 清洗前 | 保留 | 隔离 |
| --- | ---: | ---: | ---: |
| 选择题训练 | 4500 | 3813 | 687 |
| 合成定位训练 | 1800 | 1800 | 0 |
| 合成定位验证 | 200 | 200 | 0 |
| 旧选择题测试 | 217 | 152 | 65 |

合并训练集为 **5613 条**，包含 MME 1974、XLRS 1839、合成定位 1800。
保留的样本中 **1113 条标签发生修订**。286 份逐题修复候选都通过独立静态/生产桥验证；
仍带未决风险的修复候选没有强行进入严格训练版。
保留样本的修订类型计数（可重叠）：`{"system_prompt_alignment": 2228, "reviewed_semantic_repair": 264, "canonical_rank_alias": 102, "restore_count_conditions": 296, "explicit_bbox_size": 527}`。

687 条训练隔离并不全部是错误标签。其中 277 条与原留出集出现相同规范化问题文本，
涉及 12 种问题；这是严格的“问题文本不跨集合”评估策略，不能据此断言同图泄漏。
同一道通用问题用于不同图片可能合理，这部分可按后续评估方案从队列恢复。
其余隔离原因包括条件缺失、候选属性无依据、错误分组单位、序数/基数歧义等。
各原因可能重叠，逐条以 `decisions.jsonl` 和 `quarantine.jsonl` 为准。

## 实质性修复

- 恢复题目完整条件，避免“停车场内的车”被简化为统计整个图像角落。
- 修复 EntitySet 计数中的无效属性过滤：COUNT 只读集合长度，颜色/状态条件必须在上游生效。
- 修复单体/集合混淆、参考物方位被当成整图裁剪、目标与背景修饰关系错误。
- 清理 GROUP 后按行/列选择的假设：生产 GROUP 保留原成员，ORDINAL/COUNT 不自动消费 group_id。
- 修复把对象内部角落解释成外侧 SUBREGION、nearest 被替换为 NEAR、屋顶面积被替换成建筑面积等标签。
- 将合法面积排序别名规范为 bbox_area；补齐题目明确提供的 bbox 图像尺寸。
- 统一旧/新样本的当前系统提示词，避免同一训练批次使用不含 SELECT_SEM 的旧契约。

自动修复仅作用于受限、可保留完整原题条件的模板。复杂分组、序数、参考物、
双图计数和不清楚的目标类别不通过通用关键词规则强行重写。
人工审核还纠正了两处误报：合法全角问号不等于编码损坏；已有完整 bottom-row
SEM+COUNT 的计划不能仅因 GROUP 不适用而隔离。原审核证据及独立裁定均保留。

## 覆盖与局限

全部 6717 条有标签源样本执行了规则扫描。旧版本和课程重复曝光都映射到此规范源集合，
没有把重复曝光算成独立数据。人工审核报告区分模板族扫描、逐题复核与历史转换审核：
旧 MME 覆盖 1373，新 MME 覆盖 850，XLRS 当前/历史人工审核并集 928。
这些数字不等于所有样本已经逐图证明正确。旧 MME 报告说明了模板扫描和人工复核的范围；
其中所有已提供修复候选均由主任务重新绑定到源问题，并通过静态/生产桥复核，
但没有对全部图像逐条人工标注。
旧的 80 条纯英文问题也检查了非空、唯一性及字符完整性，另存 `questions_only_audited.jsonl`；
它们没有 Gold DSL，未混入 SFT。

词汇审查是高召回风险筛查，可能误报，也可能漏掉复杂指代反转或物体类别错误。
没有读取原图/答案，没有运行真实 Planner/VLM，也没有验证检测召回、单体唯一性或视觉可行性。
**624 项测试通过和静态合法不代表真实模型准确率已经提升。**

## 验证结果

- 回归：`624 passed`，见 `regression_tests.txt`；新增 18 条清洗测试。
- 最终保留 5965 条样本重新通过 parser、问题相关 Validator、
  JSON/DSL 往返、约束解码完整程序检查及实际生产桥。
- 原始源文件 SHA256 不变，原 217 条测试文件按字节保存在 `original_test_reference.jsonl`。
- 清洗后的训练与留出集 ID/规范化问题交集为 0；隔离与训练 ID 交集为 0。
- 本地 4B tokenizer 校验全部 5613 条训练 token 前缀和完整 assistant 后缀；
  对各批最长样本共 64 条执行真实训练 collator loss-mask 校验。
- 完整序列最大 1472 tokens，
  assistant 最大 223 tokens；
  433 条超过原 1088 长度，
  0 条超过输出 512。
  建议 `max_seq_length=1536`、`max_new_tokens=512`，
  训练配置未自动修改。
- 源文件一致性/数据隔离见 `release_integrity.json`，前后真实例子见 `before_after_examples.json`。

## 使用与代价

`choice_train.jsonl` 是严格清洗选择题集；`locate_train.jsonl` 是定位训练；
`train.jsonl` 为二者无重复曝光的直接合并，定位占比约 32.1%，不沿用旧课程曝光或强制 20% 配比。
训练时应显式决定混合比例，避免误把本次清洗当成已经重新训练。
`test.jsonl` 是清洗子集，不能把它的分数直接与原 217 题分数比较。
新增完整候选筛选通常会多一次 SEM VLM 调用，并可能触发已有候选预算下的未决状态；
本轮没有修改候选预算、显存参数或后端策略，运行耗时和准确率需真实模型实验测量。

## 新增文件

- `quality/planner_cleaning.py`：真实执行契约审查、受限计数条件修复、bbox/别名修复。
- `tools/clean_planner_dataset.py`：分片扫描、人工修复源绑定、严格发布、隔离与原文件保护。
- `tools/verify_clean_planner_sft.py`：选择题/BBOX 混合数据的 tokenizer/实际 collator 校验。
- `tools/finalize_planner_cleaning.py`：后台补齐验证、汇总证据与本报告。
- `tests/test_planner_cleaning.py`：18 条语义与发布集成边界测试。
- 本目录的审核、独立修复验证、源覆盖、前后对照和日志文件。

## 可复现命令

使用指定 Python，设置 `PYTHONPATH=src;.`。先按 `--offset/--limit`（每批最多 1000）
运行 `python -m taskgraph_lab.tools.clean_planner_dataset audit`，随后 `release`。
现有正式输出不可覆盖；复做新版本应选择独立目录。
编码使用 `python -m taskgraph_lab.tools.verify_clean_planner_sft --input ... --tokenizer ...
--output ... --offset N --limit 1500`。耗时超过 30 秒的收尾应按仓库要求后台启动。
回归命令：`python -m pytest -q --basetemp <独立临时目录> taskgraph_lab/tests
tests/unit/taskgraph/test_planner_provider.py tests/unit/taskgraph/test_locate_select_v2.py
tests/unit/taskgraph/test_locate_routing_v2.py`。

训练数据 SHA256：`2c0c28788421dbf9cb3a866b25e21db430c656d7aabbe61effa6204838e0d49c`。
