# LOCATE / SELECT v2：修改后提示词审计

本报告审计实际实现，不把静态测试当准确率实验。完整源码、SHA256、真实构造的提示词（包含 Qwen chat template、图像 manifest、结构化上下文、options、reasoning instruction 与评分 suffix）位于 [修改后快照](locate_select_v2_after/source_sha256.json)、[完整渲染](locate_select_v2_after/rendered_prompts.md)、[机器记录](locate_select_v2_after/rendered_prompts.json)。所有示例使用 synthetic scene、mock DINO 和捕获模型；图和提示词构造实际运行，未加载权重。

修改后14场景、19次捕获调用，包括旧整图choice包装、REL cached scorer不可用时的token约束fallback、默认关闭的legacy生成入口。当前函数精确行范围见 [callsites.json](locate_select_v2_after/callsites.json)；以该次源码SHA256快照为准。

## 实际调用与可达性

| 入口与触发 | 提示词字段和视觉材料 | 输出、缓存与失败 |
|---|---|---|
| `LocateExecutor.execute` → `ReferentRefiner.vlm_referent`；图 hint=SINGLE_TARGET | `Task: LOCATE_SINGLE... Target category: {target.category} Intrinsic attributes: {stable JSON} Actual supplied visual size: {w} x {h}...`；全文见渲染。只当前 LOCATE TargetSpec；兼容签名 question 完全不用。ImageRef/局部 Region 均 compose_locate 单张 LOCATE_SCOPE；无候选框、无原问题、无下游 SELECT.query | `infer(locate_single)`，单阶段 JSON自由生成、无 KV跨调用复用或 mask；严格 status+字段集合解析。OK 需有限、有序、界内 bbox，render面积≥4px²；逆 scale_xy+crop_origin映射，再检查原图与scope界限。NONE→EMPTY、AMBIGUOUS/UNRESOLVED→0实体明确状态。非法输出不重排、不 clamp、不造scope代理；配置 single_detector_fallback=true 时只对 UNRESOLVED 回退DINO |
| `LocateExecutor` 单体DINO回退且>1有效候选 → bounded selector | 仅 `Identify the unique target defined ONLY by {TargetSpec JSON}`，purpose=locate_single_fallback。下游query和整题不进入。既有候选红A/B、参考物蓝REF（如有），另有scope overview；每批≤8，单次总像素≤409600，stride32且每视觉≥65536 | 同 SELECT_SEM_SINGLE 的有限 `A..H/NONE/AMBIGUOUS/UNRESOLVED`；有推理再有限续写评分、每次finally释放cache。跨批多正项→AMBIGUOUS；缺映射、异常、不可见候选、超64预算→UNRESOLVED；DINO空且原VLM未决仍未决 |
| `LocateExecutor` SET_REQUIRED → DINO →可选固有属性核验 | 检测类别 query=TargetSpec.detection_phrase；CLIP top-K 不参与该集合。attribute_policy 默认 unresolved，保留未验证候选并阻止精确集合消费者；real.example 显式 vlm 时用 `Match the category and ALL intrinsic attributes of this LOCATE TargetSpec: {JSON}`；purpose=locate_attribute_verification，包含类别及全部固有属性，背景不额外检测 | 与 SEM_MULTI 同三态有限评分，每候选 MATCH/NO_MATCH/UNCERTAIN；最多64串行调用、≤8可见候选/批，总像素409600；拒绝不确定当negative，不确定候选保留作audit且pool incomplete。属性核验不做singleton消歧；没有未校准CLIP相似度→属性真值转换。阈值/NMS定义下的检测候选池完整并不承诺真实世界100%召回 |
| `SelectExecutor.execute` mode=SEMANTIC/SINGLE → `BoundedSemanticSelector` | `Task: SELECT_SEM_SINGLE... Select ONLY from the existing marked candidates... Complete query (verbatim): {params.query}`；不读原题。candidate/ref画布+必要scope overview；stable候选ID、原图bbox、canvas/render映射均可追踪。query中的树/草/道路只是上下文，无额外LOCATE | 每批 `FiniteDecisionRequest(SINGLE,purpose=select_semantic)`，自由推理后同KV打分 `A.../NONE/AMBIGUOUS/UNRESOLVED`；suffix=`\n\nSELECT_SEM_SINGLE decision:`。SINGLE跨批唯一→OK、全NONE→EMPTY、多正或AMBIGUOUS→AMBIGUOUS、未决或超预算→UNRESOLVED；sentinels从不映射为实体。无自由文本ID解析回退 |
| `SelectExecutor` SEMANTIC/MULTI | `Task: SELECT_SEM_MULTI... Complete query (verbatim): {params.query}`；完整复合描述不缩成关系枚举。视觉与SINGLE同上 | 每候选独立 `FiniteDecisionRequest(SINGLE)` 候选值固定 `MATCH/NO_MATCH/UNCERTAIN`，reasoning instruction `Verify ONLY candidate {label} ... three-state finite decision...`，suffix=`\n\nCandidate {label} semantic verdict:`。这是实际三态categorical scoring，不借用CHOICE_MULTI的YES/NO。每次推理KV复用到自己的三态决策；不同候选重新prefill，未跨候选共享cache。全NO_MATCH→合法空集；任何uncertain/unseen→UNRESOLVED并保留已确认正项 |
| 原 `SelectExecutor._semantic_select` SELECT_REL | 原关系枚举prompt、A/B/REF、geometry clear/grey partition 均保留；只有grey送VLM，clear结果合并。purpose=select_relation；完整自然语言仅由新SEM消费 | 原CHOICE_MULTI YES/NO、原阈值0、原 KV缓存后缀不变；原受约束≤8 label子集 fallback仍在。grey>batch_size默认8时，在绘制/调用前明确UNRESOLVED，列全部unseen IDs；不再让海量候选产生OOM或悄悄Top-K。确定性关系不受此视觉预算限制 |
| 原 `ReferentRefiner.refine` / `select_grid_ids` | 类与旧可调用入口保留，模板包含 Original referring question/expression；完整模板与修改前相同，after渲染再次实际构造。real.example的semantic_weight=0仍会跳过refine VLM，但不影响grid入口 | **新的 LOCATE 执行链不再调用这些旧方法**。仅外部/旧直接入口及单元测试仍可达。旧预算/margin/geometry fallback规则保留用于兼容；不作为新候选集合的过滤或成功定位来源。不能把此旧入口的结果当新执行链的验证结论 |
| `ReferentRefiner.visual_fallback` | 无prompt、无VLM调用 | 现在0实体、UNRESOLVED、pool_complete=false；不再伪造scope目标 |
| `ChoiceResolver._semantic_fallback_score` 保留入口 | 原 bounded candidates+原始final残余question+options+CHOICE_MAP，完整prompt附于after渲染。未改该文件 | 原purpose semantic_candidate_fallback、默认generic reasoning、原KV/finite choice后缀；仍供旧外部入口用。新 incomplete pool 在 GraphExecutor 及最终证据边界拒绝，不进入此路径 |
| 原最终 whole-image choice fallback | 旧图旧失败路径保留，以兼容既有接口行为 | 对新 incomplete/unverified池及 SEM非OK最终证据抛CandidateIntegrityError，阻止回整图重新回答。defer_final_answer允许新SEM合法EMPTY透传；COUNT可消费SEM_EMPTY得0。无HTTP或最终序列化字段修改 |

## 公共包装与模型消息

`LazyQwenSemanticProvider._prompt` 顺序不变：Visual inputs → Structured results → Question → optional Options → reasoning_instruction。engine._build_messages 为单条 user，多图先、文本后；没有另外隐藏的system instruction。仓库本地 Qwen3-VL-2B-Instruct/chat_template.json 的实际 Jinja 渲染保存在所有调用记录。placeholder 由 processor 换成实际 image tokens；本次未运行 processor/权重，所以不伪造视觉 token IDs。新LOCATE与SEM渲染尺寸为32倍数，满足本地Qwen3-VL patch16/merge2、minimum65536像素约束，以减少隐式processor resize；不同模型/processor配置须重新验证此契约。

`reason_and_decide` 复用原 HF engine.reason_and_choose：SINGLE（含MATCH三态）生成reasoning后缓存续写评分合法多token字符串，后缀显式传入，finally关闭session。MULTI旧REL保留每选项复制reasoning cache再YES/NO评分。新SEM不解析reasoning正文、不使用未约束生成选择IDs；无cached provider或非法score/ID/mapping均显式未决。并行加载新模型、并行GPU推理、多框VLM LOCATE均未引入。

预算同时取provider.model_config.max_pixels上限；单体按单视觉限制，新SEM按两视觉限制和总预算的较小值渲染。selected ID与最高分不一致、非有限分数或终端平分均未决。原REL受约束fallback的合法字符串集合完整保存在allowed_outputs；旧最终生成fallback仅provider没有cached评分方法且legacy_regex_fallback=true时可达，cached方法自身抛异常不会自动走该legacy入口。

不完整池由worker/sidecar/tile/adapter向上传递；新语义空集和单体LOCATE NONE不能触发单体属性猜答，嵌套SELECT保留lineage。defer_final_answer允许语义合法空集，COUNT可得0；非defer最终语义非OK证据不回整图。

## 五个完整图的执行证据

全部图均经现有DSL compiler/parser/Validator往返，并实际运行GraphExecutor；逻辑图执行前后相等。记录最终source值，示例3保留合法CHOICE_SINGLE残余问题，但本轮只执行选择source，不进行最终benchmark选项评分。

| 示例 | 物理路由与实际mock结果 | 记录 |
|---|---|---|
| 白色方形建筑→ATTRIBUTE | VLM_SINGLE bbox `[100,60,200,140]`；ATTRIBUTE mock=white | [图、trace、prompt](locate_select_v2_after/e2e_01.json) |
| building→TOPMOST→ATTRIBUTE | DINO20候选，第20个低分框y=5被选中；mock属性white | [执行记录](locate_select_v2_after/e2e_02.json) |
| building→SEM_SINGLE完整树/草query | DINO3候选，有限输出B→candidate_0002原图框`[180,160,250,220]` | [执行记录](locate_select_v2_after/e2e_03.json) |
| red cars→SEM_MULTI→COUNT | DINO3→固有属性mock MATCH/NO_MATCH/MATCH保留2→语义mock MATCH/NO_MATCH保留1→cardinality=1，无再检测 | [执行记录](locate_select_v2_after/e2e_04.json) |
| building参考物+cars→REL→COUNT | reference VLM_SINGLE；cars DINO20；确定性LEFT_OF选10；COUNT=10 | [执行记录](locate_select_v2_after/e2e_05.json) |

这里每个预测都有mock来源标识，不能据此报告模型准确率提升或真实显存峰值。
