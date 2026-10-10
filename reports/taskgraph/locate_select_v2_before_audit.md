# LOCATE / SELECT v2：修改前提示词审计

2026-10-10。分支 `codex/competition-protocol-v2`，HEAD `7ee24b5`。工作树原有 competition、hash memory、HF KV cache 等修改，审计以当前工作树为准，未重置。正式执行代码修改前保存本报告与源码快照。

完整业务模板及行号见 [源码快照](locate_select_v2_before/source_sha256.json) 所列文件；不是只审查业务短提示。五个实际构造的调用示例，含公共包装、实际 Qwen chat template、图像角色、选项、推理指令、所有决策后缀及终端候选，见 [完整渲染](locate_select_v2_before/rendered_prompts.md)、[机器可读记录](locate_select_v2_before/rendered_prompts.json)。这些调用实际执行原提示词构造与 InputComposer，图像为合成审计场景、模型为捕获 mock，不代表真实推理准确率。快照保留修改前函数的完整源码，防止后续行号漂移掩盖差异。

## 逐路径核验

| 调用及源行 | 触发与配置可达性 | 模板与变量来源、视觉 | 决策、缓存、解析及失败 |
|---|---|---|---|
| `ReferentRefiner.vlm_referent` 580–689；`LocateExecutor.execute` 1074 | refiner 存在且 enabled、singleton_vlm_primary，旧 `_needs_refinement` 判单体；类默认 enabled=false、singleton=false；real.example enabled=true 但 singleton 未开启，因此部署示例不走此调用 | `Locate the target ... pixel coordinates ... Target: {target.phrase()} Original question: {question} Output format: x1 y1 x2 y2 ... Return only the four numbers.` target 来自当前 LOCATE；question 来自整题 context；`compose([image])` RESULT_1 全图，无候选标记；局部 Region 直接拒绝 | `infer(text)` 单阶段自由生成，无 KV 复用/有限评分/token mask。正则取首四数，反序重排并 clamp，面积 ≥ 原图 0.0001。错误/无框返回 None；存在缩略图坐标不明确和下游条件泄漏风险 |
| `ReferentRefiner.refine` 342–579 | `_needs_refinement` 且候选非空；候选>1、enabled 且 semantic_weight>0 才真正调用。real.example semantic_weight=0，因此当前示例只取 detector/CLIP 最高分，跳过 VLM | `You are verifying which supplied candidate matches the referring expression ... Original referring question: {question} Target specification: category={...}; attributes={...}`，question 为整题；预算预排最多8候选，compose_named candidates，A/B/... 红框，union+halo 画布，options Candidate A: stable ID | `reason_and_choose(CHOICE_SINGLE,purpose=referent_refinement)`，推理后同 KV 有限 label 评分；后缀 Final choice。semantic minmax 分数与整题空间先验融合；margin≤.02 或异常保留预算子集并 fallback_required。没有 NONE 终端候选；最高 detector 分并非属性验证 |
| `ReferentRefiner.select_grid_ids` 176–312；`LocateExecutor._regional_fallback` 925–1059 | 单体检测为空时区域回退；enabled 时 VLM，未启用/失败时整题空间关键词选网格；semantic_weight=0 不会禁用此 VLM 调用 | `Select one to three grid cells ... Original referring expression: {question} Target specification: ...`；3×3 RegionSet→A..I；选项 Grid A: top-left 等；输入整题与 target；最多前三网格再次检测 | CHOICE_MULTI/purpose locate_grid_fallback；一次推理 KV，每网格 YES/NO 后缀；阈值0；无正项则强选最高分，最多3；异常几何回退；选区覆盖不足但可报 resolved 是风险 |
| `ReferentRefiner.visual_fallback` 690–780 | 检测+区域回退均空，或 retriever 与 detector 均空，且 refiner enabled | **没有 VLM 调用，没有 prompt**。直接把 ImageRef 全图或 Region 当一个 Entity | score=0、fallback_required=true，却标 SEMANTIC_FALLBACK_RESOLVED；此代理框不能当目标定位成功 |
| `SelectExecutor._semantic_select` 1979–2240 | SELECT_REL 几何灰区/语义关系；原 RELATION 模式，SINGLE 多几何正项直接 AMBIGUOUS；其他走语义。没有 SEMANTIC 模式 | `Analyze which candidate objects satisfy relation {relation} relative to the marked reference ...` 仅 relation 枚举，不能表达完整自然语言条件。compose_named grey candidates+reference，红 A/B、蓝 REF，union+halo，无额外整图；candidate_mapping 对齐 stable ID。option_texts 在评分后缀，业务 compose 未附 options | reason_and_choose CHOICE_MULTI/purpose select_relation；同 KV 独立 binary 验证，阈值 choice.multi_select_threshold 默认0；clear positives 并入。SINGLE 多正项 AMBIGUOUS、0项 EMPTY。只有 CachedChoiceUnavailableError 触发 infer(selection) 备用；≤8 label token mask 允许 NONE/任意子集，未受约束/无效输出→UNRESOLVED；>8不降级自由解析。其他异常向上抛 |
| `LazyQwenSemanticProvider._prompt` 1653–1671；`infer` 1672–1718 | 上述所有调用公共包装；无单独 VLM 系统指令 | 顺序 `Visual inputs: [image_1] role ...` → `Structured results` → `Question` → optional `Options` → optional reasoning instruction。InputComposer 来源角色、候选/REF manifest、结构化 region/entity 信息均包含。engine._build_messages 把图像按顺序放 user.content 前部，text 最后；无 system message；本地 chat_template.json 精确渲染保存在示例 | infer 仅 selection 且 selection_constrained_decoding=true、1..8单字母label 时枚举合法字符串 token mask；text 无 mask。没有视觉时 text-only。模板视觉 placeholder 后由 processor 展成实际 image tokens；未加载权重的审计不伪造这些 token IDs |
| `reason_and_choose` 1777；`reason_and_decide` 1738；engine 1133–1295 | 实际有限决策统一路径；缺 engine cached scorer→CachedChoiceUnavailableError | choice purpose=select_relation 时指令 `Analyze which candidate object or objects satisfy the requested relation ... A separate constrained step ...`；其他 referent/grid 默认 `Analyze the visual evidence, question, and all candidate options carefully ...`。完整文本在渲染记录 | 两阶段 reason_with_cache → score_choice_from_cache；SINGLE `\n\nFinal choice:`、canonical `\n\nCanonical value:`；MULTI `Verify the following option independently.\n\nCandidate option {choice_id}: {option_text}\nIs this option a correct answer to the original question?\nAnswer YES or NO:`。SINGLE legal IDs评分并消费cache；MULTI每项复制cache评分YES/NO差>0；finally关闭session。是有限续写评分，不解析自由推理正文；semantic_decision uncertainty_epsilon默认0 |

## 间接路径与 CLIP

补充核验：初稿未单独列出以下最终证据回退。补记时 `choice.py` 的工作树与 HEAD 一致，函数没有被本轮修改；因此下面是修改前仍适用的原始模板，并非把新模板当作修改前记录。修改后渲染集还实际构造了此保留入口。此处披露审计遗漏，不声称它已在首次编码前逐项列出。

| 保留的相关调用 | 完整业务模板、变量及视觉 | 包装、评分与失败 |
|---|---|---|
| `ChoiceResolver._semantic_fallback_score` 313–366 | 单个 EntitySet 有 fallback_required、UNRESOLVED 或 SEMANTIC_FALLBACK_RESOLVED；`Use only the supplied bounded visual candidates to resolve the remaining referring expression and answer the residual question. Do not search outside the supplied candidates and do not infer a candidate that is not shown.\n\nOriginal question: {request.question or ''}\nThe candidate resolver could not establish a reliable singleton. Choose the answer option supported by the supplied candidates.`；compose_named candidates，A/B画布、原始 options 及 CHOICE_MAP；request.question 来自 final 残余问题 | `reason_and_choose` purpose=semantic_candidate_fallback；默认 generic reasoning instruction、single Final choice 或每原始选项 YES/NO suffix；复用推理 KV。缺缓存评分抛异常；输出最终 choice，不返回新的 Entity，不是 LOCATE。旧 refiner.refine 的未决结果可达此路径 |
| `TaskGraphRuntime._taskgraph` 原 final evidence fallback；`ChoiceResolver._cached_score` | SELECT EMPTY/UNRESOLVED 使 unwrap 或 composition 产生 ValueError 时，旧代码把整个 images 放回最终 choice，question=graph.final.question or request.question；compose 全图 + options + CHOICE_MAP；业务 question 未额外裁剪；有系统性绕过失败证据风险 | 同一 Qwen 包装、generic reasoning instruction、有限 choice IDs/YES-NO 评分与 KV 复用；choice.legacy_regex_fallback 默认false，显式true且无cached scorer才 infer(choice) exact ID/text parser。该失败回退在新候选完整性状态上必须阻止，最终协议字段本身不能改变 |

`LocateExecutor._regional_fallback` 内 select_grid_ids 是唯一间接 LOCATE VLM；检测/retriever/CLIP 本身不调用 Qwen。旧 visual_fallback 后续 ATTRIBUTE/CLASSIFY/RELATION/最终 choice 属原图已有独立语义节点，不是该函数的隐含 VLM；它们的运行时语义模板完整保存在 semantic_prompts.py 和 operators.py 源码快照中。

ProposalDetectionAdapter：TargetSpec.detection_phrase 为类别+同义词（attributes_sent_to_detector=false）；CLIP ranking_query=target.phrase；`use_clip_rerank` 与 `_needs_refinement` 耦合；集合消费者直接导致 false，固有属性被跳过。CLIP minmax融合排序与 Top-K 不是属性真值，也不是概率。当前 TargetCapabilityClassifier 按类别将 open vocabulary 导向 retriever；本轮将由图职责取代此物理选择。

旧 `_needs_refinement` 815–850：SELECT/GROUP/COUNT/RELATION 直接消费者优先 false；直接 FINAL 用整题 count 关键词；target.attrs 和单体消费者才 true。未按 input role 区分 candidates/reference，也没有有向间接集合分析。旧 execute 1331 全局 Top16 截断会丢失极值、序数、排名、分组与计数候选。InputComposer.entity_set_max_crops 只是视觉预算，不能修复逻辑截断。

## 快照与示例范围

五个示例：白色方形建筑单体定位、检测后单体 refinement、空检测3×3回退、REL NEAR SINGLE、REL NEAR MULTI。均从当前函数实际构造，不手写替代其模板。渲染包含最终评分后缀，reasoning 正文用 `AUDIT MOCK reasoning` 占位，不能在没有真实模型调用时声称是模型实际生成内容。修改后另提供完整合法 TaskGraph 执行的五条验收链。
