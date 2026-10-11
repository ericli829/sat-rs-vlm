# 4B Planner 与 2B ChangeChat LoRA 训练方案

日期：2026-10-11。状态：4B Planner 已在云端启动一轮续训；2B ChangeChat 两路线仍待后续机器/任务安排。本文件同时记录训练方案和已启动任务信息。

## 1. 本地核对结果

### 新 Planner 数据

- 数据发布目录：`taskgraph_lab/data/planner_quality_clean_v1/`。
- `train.jsonl` 共 5,613 条：选择题 3,813 条、合成定位/BBOX 1,800 条；BBOX 占 32.1%。
- 另有选择题测试 152 条、合成定位验证 200 条。原始 217 条历史测试作为参考保存，不能直接与清洗后的 152 条子集比较总分。
- 前一轮发布验收：624 项回归测试通过；保留的训练/测试/验证样本通过语法、类型、约束语法和生产桥接检查。这不代表模型预测已经准确。
- 当前 4B tokenizer 下完整输入最长 1,472 token，答案最长 223 token；旧上限 1,088 会截断 433 条训练样本，1,536 下无截断。
- 训练文件 SHA256：`2c0c28788421dbf9cb3a866b25e21db430c656d7aabbe61effa6204838e0d49c`。
- 现有 4B 起点已找到：`D:\Desktop\tzb-2026\final models\qwen3vl_4b_planner_hard_refine_v1_20260902_234235`，包括 adapter 权重和元数据；LoRA rank=16、alpha=32。

### ChangeChat 数据

本地源：`F:\VIT-data\changechat-105k`。该目录只有标注；图片可复用 `E:\迅雷下载\LEVIR-CC\images`。实际训练标注引用的 13,630 个 A/B 路径均存在；抽查首对图像均为 RGB、256×256。这是路径存在性检查及一对图片头检查，尚未逐图验证解码或内容去重。

| 训练任务 | 原始对话条数 |
| --- | ---: |
| 变化描述 | 34,075 |
| 是否变化 | 6,815 |
| 变化计数 | 6,815 |
| 九宫格变化定位 | 6,815 |
| 开放问答 | 26,600 |
| 三轮问答（六条消息） | 6,815 |
| 合计 | 87,935 |

以上任务统计来自实际训练文件中每个图像对的记录结构，已核对全部 6,815 组顺序；训练文件没有显式 `type` 字段，转换时必须补充来源和任务元数据，不能照搬测试文件的类型字段。

- 两个训练 JSON 均有 87,935 条、6,815 个图像对。
- `train` 与 `train_loc` 有 81,120 条完全相同；6,815 条差异全部是定位答案的文字包装。解析全部定位答案后，道路和建筑的九宫格集合完全一致，未发现非法网格名称或解析失败。二者选一个作为源，不能拼接扩充样本量；建议使用更简洁的 `train.json` 对应版本，即 `changechat_105k_train.json`。
- 单个训练版本忽略记录 ID 后发现 55 条完全重复对话，转换时去重并报告。
- 训练集 81,120 条单轮对话、6,815 条三轮问答；结构检查通过。
- 六个测试文件均覆盖相同的 1,929 个图像对：描述、二分类、计数、定位、多轮各 1,929 条，开放问答 7,527 条。多个任务不能当作独立图像泛化样本累加。
- 训练与测试按文件路径比较，图像对交集为零；尚未进行图片内容 SHA256 去重。
- 原始图片目录实际有 train=7,590、val=1,438、test=2,135 对；标注只使用其中子集。必须按标注引用建清单，不能根据 README 或最大编号猜样本数。
- 本地 replay 起点已找到：`D:\Desktop\tzb-2026\results\levir_train\vrsbench_levircc_replay_formal\round_2_adapter`，包括权重与元数据；LoRA rank=16、alpha=32，基座为 Qwen3-VL-2B-Instruct。原始 VRSBench adapter 需在机器就绪后按训练血缘核实具体目录和权重。

官方资料：[数据卡](https://huggingface.co/datasets/hlwu/changechat-105k)、[DeltaVLM 作者代码](https://github.com/hanlinwu/DeltaVLM)。这里是 DeltaVLM 配套的 ChangeChat-105k，不能与旧版 ChangeChat 的数据规模混淆；本方案的数量以本地 JSON 实测为准。

## 2. 4B Planner 训练

目标是让模型学会清洗后的 DSL 和语义约定，特别是 SELECT_SEM 的完整 query、reference/scope 的独立角色、SINGLE/MULTI、COUNT 候选消费和 BBOX 最终输出；不增加图像输入，不改变 Planner 职责。

1. 从当前 hard-refine 4B adapter 继续训练，使用同一 Qwen3-VL-4B-Instruct 基座及 tokenizer。加载可训练 adapter，重新建立 optimizer/scheduler；不恢复旧 Trainer 状态。
2. 使用 5,613 条作为训练候选池；从 3,813 条选择题中按规范化问题分组、兼顾来源及操作类型留出约 10% 作开发集。原 200 条定位验证作为开发集的另一部分，152 条选择题测试封存。最终训练量约 5.2k，精确数量以分组切分结果为准。
3. 同义/重复问题必须同组；报告开发集是否与训练共享题型模板，不能把模板内验证包装成模板外泛化。没有可靠图像分组元数据时明确记录这一限制。
4. 保留混合数据的自然分布作为第一轮基线，不复用旧 hard-curriculum 的重复曝光权重。分别报告选择题、BBOX 和 SELECT_SEM 子集，避免被合成定位样本的成绩掩盖选择题退化。
5. 建议 LoRA r=16/alpha=32，冻结视觉编码器及投影模块，BF16、gradient checkpointing；`max_seq_length=1536`，生成预算 512。学习率先取 `2e-5`，有效 batch=16，warmup=5%，weight decay=0.01。micro-batch 依据云端实测调整。
6. 先训练 1 epoch 并评估，再视开发集走势追加至总计 2–3 epoch；每约 100 optimizer steps 保存并评估。约 5.2k 样本、有效 batch 16 时每轮约 330 steps，不能在机器实测前承诺时长。
7. 同时测不受约束输出和正式约束解码：语法/类型通过率、最终输出协议、条件与目标保留、SELECT_SEM query/scope/reference、候选集合与 cardinality。精确 DSL 匹配只作辅助，因为一个问题可能有多条等价合法路径。
8. 与训练前 4B 在同一开发集比较，选定 checkpoint 后再测封存测试；真实图像上的端到端收益另行验证。

## 3. 2B ChangeChat 九宫格定位优先方案

本轮 2B 的首要目标改为赛题口径的 LEVIR-CC 九宫格变化定位。训练和选模都以道路/建筑九宫格集合为中心；描述、二分类、计数和 VRSBench 回放只作辅助监督与能力保持。该监督是离散网格集合，不是连续框，不以此宣称提升通用 bbox 定位。

### 两条初始化路线

| 路线 | 初始化 | 主要意义 | 需要观察 |
| --- | --- | --- | --- |
| A | 第一版 VRSBench LoRA | 检查模型在没有旧 LEVIR-CC caption 微调的起点上能否学好九宫格任务 | 九宫格学习速度、变化召回、对 VRS 能力的遗忘 |
| B | 现有 replay round-2 LoRA | 检查已有 LEVIR-CC 描述训练是否有帮助；不能假定 caption 能力等于九宫格能力 | 旧任务偏好是否妨碍网格集合预测，以及是否过拟合描述习惯 |

两条路线使用同一基座、同一清洗数据、相同图像对分组、采样清单、随机种子、学习率及 optimizer steps。分别在各自 adapter 上继续训练，不串联 A→B，不叠加两个 adapter，也不沿用旧 optimizer。先分别测训练前九宫格基线，再比较训练后官方口径主指标及各自增量；不能只比较 loss。还需核对旧 replay 的训练清单/图像内容哈希，确认两条路线的图像曝光差异，避免把来源差异误判成初始化效果。

### 数据转换与必要检查

- 保留两张独立图片的 A→B 顺序，在首个 user 消息中放置两个图像项并说明前后时相。保持原始细节，交由 Qwen processor 合法调整尺寸；不先拼成单张图或大幅缩小。
- 转换为项目 `messages` JSONL，保留原始来源 ID 和图像对 ID。第一轮使用单轮九宫格及单轮辅助任务，暂不纳入多轮记录。
- 当前 `Qwen3VLDataCollator` 会删除所有 assistant 消息生成 prompt，再按总长度一次屏蔽前缀。该方法适合单轮，不能正确监督六条消息的多轮对话；生成模式也会删除历史 assistant。若后续想纳入多轮，先实现逐 assistant 片段的监督 mask 和生成历史保留，并验证各 assistant、user/image/padding mask。当前只是源码审计，尚未实现修复。
- `train.json` 的 6,815 条九宫格答案均可解析，无非法网格标签。按答案集合统计：道路和建筑都为空 3,522 对（51.7%）；至少一类非空 3,293 对（48.3%）。四个互斥子类为双空 3,522、仅道路 562、仅建筑 957、道路与建筑均非空 1,774。道路集合为空 4,479 条，建筑集合为空 4,084 条。官方格式 `test_loc` 先前核对为 1,929 对，双空/有目标分别 965/964；训练和测试文件路径没有交集，仍需在内容级去重后封存测试集。
- `changeflag` 不在九宫格记录本身提供，但可按图像对与同文件其他任务记录关联。关联后的统计为：`changeflag=0` 且网格双空 3,257；`changeflag=0` 但网格非空 151；`changeflag=1` 但网格双空 265；两者都为正 3,142。全局变化可能发生在道路/建筑以外，因此 265 例不必然是冲突；151 例是重点复核的潜在标签矛盾。清洗时保留原始标注和复核状态，先复核这 151 对；不要用 `changeflag` 覆盖九宫格答案，也不要将其当作九宫格正负标签来源。
- `train` 与 `train_loc` 的九宫格集合完全一致；只选一个来源，采用答案更简洁的 `changechat_105k_train.json`，绝不拼接两份。55 条完全重复的非定位对话按内容去重。训练记录没有可靠 `type` 字段，转换时依据 prompt/来源显式生成任务元数据。
- 按图像对分组，从 6,815 个训练图像对中按“双空、仅道路、仅建筑、两类都有”分层留出约 10%（约 682 对）作九宫格开发集，同一图像对的所有任务/轮次一并留出；两路线共享完全相同的 split。剩余约 6,133 对作为训练池。ChangeChat 的 1,929 对 `test_loc` 封存，只在路线和 checkpoint 选定后作最终评测，不能用它反复调参。
- 九宫格输出严格监督变化区域所在的九格集合，不转换成精确 bbox。对答案空集、道路/建筑独立输出和双时相次序做格式与标签审查；不确定样本隔离。
- ChangeChat 的 87,935 条训练记录中，九宫格只有 6,815 条（7.75%）；如果把所有记录均匀混训，目标会被五条 caption/图像对、多条开放 QA 和多轮回答稀释。因此必须按目标曝光率采样，不能直接均匀遍历原始 JSONL。
- 六个测试文件封存；同一图像对的所有任务及所有轮次必须同属一个 split。

### 建议训练安排

1. 先测 A/B 训练前基线，并完成 151 个潜在冲突九宫格标注的复核或隔离；固定清洗清单与图像对分层 split 后，A/B 从同样数据起点分别训练和验证。
2. 首轮把训练曝光槽位设为 **九宫格 70%、单轮辅助 ChangeChat 20%、VRSBench replay 10%**。这是样本曝光次数比例，不是原始数据占比。约 6,133 条训练网格样本各曝光一次时，总曝光约 8,761 条：网格 6,133、辅助 ChangeChat 1,752、VRS replay 876，effective batch=16 时约 548 optimizer steps。按完全相同的样本清单和 steps 对照 A/B；废止“ChangeChat:VRSBench=3:1”这一与九宫格主目标不匹配的旧建议。
3. 辅助 ChangeChat 只从非多轮单轮任务采样，首轮内部比例建议二分类 35%、计数 35%、caption 20%、开放 QA 10%，以巩固变化语义同时抑制 caption/QA 数量优势。VRS replay 按 grounding/counting/VQA/caption/scene 分层，使用固定子集及确定采样种子。ChangeChat 已含旧 LEVIR caption，第一轮不再额外重复加入 LEVIR caption。
4. 网格流按双空、仅道路、仅建筑、两类都有分层洗牌，但总体维持清洗后的分布；其空/非空比例接近 held-out test，不应为追求非空召回而人为改变比赛先验。记录每类曝光次数，保证 A/B 可复现。
5. 起步 LR=`1e-5`、effective batch=16、BF16、gradient checkpointing、r=16/alpha=32。先冻结视觉编码器和 merger，只更新语言 LoRA，核对真实可训练参数清单；旧配置的 `freeze_projector=false` 不足以证明 projector 实际参与过训练。第一轮不增加视觉解冻或新模块。
6. 先用 micro-batch=1–2 做双图显存和吞吐检查，再用梯度累积达到 effective batch 16；不照搬旧单轮 replay 的 batch=8。九宫格回答很短，序列上限应按真实双图 processor 编码统计后设置，确保图像 token 和答案都未静默截断；不能为了迁就 2,048 上限而裁掉图像细节。
7. A/B 同时比较官方九宫格指标与各自训练前增量。先完成约一轮九宫格训练池曝光（约 548 steps，精确步数依清洗数量和 effective batch 而定），根据开发集决定是否延长胜出路线；若继续训练，记录已曝光样本并避免无意重复加权。B 是否胜出由九宫格结果决定，不能仅凭已有 LEVIR caption 训练推断。

### 验收指标

- 变化二分类：Balanced Accuracy、Precision/Recall/F1，分别看误报和漏报。
- 变化计数：道路/建筑分别计算 exact match 与 MAE；限定区域或其他条件必须保留，无法可靠解析的回答单独计数。
- **首要指标：赛题方 LEVIR-CC 九宫格定位官方评测分数**，取得官方 evaluator 后按其实现复现。选 checkpoint 和 A/B 路线优先看此分数，不用 caption、QA 或 loss 代替。
- 九宫格开发集同时报告道路、建筑各自的网格集合 exact match 与 cell-level precision/recall/F1，按图像 macro-average；单列双空、仅道路、仅建筑、两类都有四类表现，特别检查空集误报和正例漏召。若官方分数定义与这些辅助指标不同，官方分数用于决策，辅助指标用于诊断。该标签不是 bbox，不用 bbox IoU 冒充精确定位评测。
- 描述：复用多参考描述指标，同时分别报告变化/不变样本及事实性抽查。
- 开放 QA/多轮：答案事实覆盖、跨轮一致性、历史依赖；评测使用模型自己的上一轮输出，不能只喂标准答案历史。
- VRS 保留：同一固定验证子集的 grounding IoU、计数 exact/MAE、VQA 等，与各自训练前基线比较；阈值和选择规则在看结果前固定。

## 4. 云端执行顺序与待确认项

机器就绪后先确认 GPU/显存、存储、原始 adapter 与基座、软件环境，再同步当前工作区快照、新数据与转换工具。当前工作区存在未提交的 Schema/Validator/提示词和数据修改，单独拉 Git HEAD 会遗漏本次依赖，必须固定代码快照及数据哈希。

4B 本轮已使用云端现存 hard-refine adapter；云端与本地 adapter 权重 SHA256 相同（`defad37c62a53cb928d8879d8f690cc55806dffee7a81a547f121c9975511570`）。运行设备为 NVIDIA RTX 4080 SUPER，32,760 MiB 显存。清洗数据、当前工作区源码快照和隔离配置位于共享盘 `/root/autodl-fs/data/sat-rs-vlm-training/planner_4b_clean_20261011/`；未覆盖云端原仓库的既有删除状态。训练参数为 1 epoch、有效 batch 16、学习率 `2e-5`、序列上限 1,536，每 175 步验证并保存。云端配置导入预检通过；训练进程开始时记录 5,613 条训练、200 条验证及每轮 351 steps。PID 和日志路径在本文件对应的训练启动回报中给出。

后续建议单 GPU 顺序：4B 训练 → 2B 九宫格清洗、split 与单轮管线验收 → A/B 九宫格基线和同预算对照 → 胜出路线按开发集结果决定是否延长 → 封存集评测与 VRS 回归。多轮管线修复不再是启动九宫格首轮训练的前置条件，因为首轮明确排除多轮样本。两个模型不同时驻留 GPU；先测 50–100 steps 的稳定吞吐与峰值显存，再估算时间/成本。

本轮独立训练输出目录为 `/root/autodl-fs/data/sat-rs-vlm-training/planner_4b_clean_20261011/output/training`，日志为 `/root/autodl-fs/data/sat-rs-vlm-training/planner_4b_clean_20261011/logs/train.log`，PID 文件为 `/root/autodl-fs/data/sat-rs-vlm-training/planner_4b_clean_20261011/logs/train.pid`。训练完成后再单独评估封存测试集并报告回归结果。
