# LOCATE / SELECT v2 前后审计

## A. 修改前

2026-10-10，分支 `codex/competition-protocol-v2`，HEAD `7ee24b5`。正式修改执行代码前保存了 [修改前审计](locate_select_v2_before_audit.md)、[实际调用的完整渲染](locate_select_v2_before/rendered_prompts.md)、[源码 SHA256](locate_select_v2_before/source_sha256.json)。原快照禁止覆盖。

流程上的不足：首次审计漏列了最终 choice 的间接候选/整图回退，编码后才依据未修改的 `choice.py` 补记。修改前文档已明确标注；不能声称补记在首次编码前已经完成。修改后已实际捕获保留入口及受约束回退。

## B. 修改后

[修改后完整审计](locate_select_v2_after_audit.md) 按入口列出触发、配置、字段来源、视觉、输出、缓存、后缀和失败状态。[完整渲染](locate_select_v2_after/rendered_prompts.md) 与 [JSON](locate_select_v2_after/rendered_prompts.json) 包含14个场景、19次捕获调用，包括Qwen chat template、公共包装、options、推理指令、有限候选、允许输出及suffix。

实际运行 InputComposer / LazyQwenSemanticProvider / HF message builder / 本地 chat template；预测为mock，推理正文为 `AUDIT MOCK reasoning`，未伪造processor视觉token。

## C. 端到端示例

| 完整执行链 | 结果与证据 |
|---|---|
| 白色方形建筑→ATTRIBUTE | VLM bbox `[100,60,200,140]`→mock white；[图/DSL/trace/prompt](locate_select_v2_after/e2e_01.json) |
| building→EXTREME(TOPMOST)→ATTRIBUTE | 20候选中低分第20项 y=5 被选→white；[记录](locate_select_v2_after/e2e_02.json) |
| building→SEM_SINGLE(完整树/草条件) | DINO3→B→candidate_0002原框`[180,160,250,220]`；[记录](locate_select_v2_after/e2e_03.json) |
| red cars→SEM_MULTI→COUNT | 类别3→属性2→语义1→COUNT=1；[记录](locate_select_v2_after/e2e_04.json) |
| cars→REL(building参考物)→COUNT | car DINO20，reference VLM1；LEFT_OF选10→COUNT=10；[记录](locate_select_v2_after/e2e_05.json) |

五图均通过DSL compiler/parser/Validator并实际执行GraphExecutor，逻辑图前后相等。第3图只执行final source，不评测benchmark选项；所有模型预测均mock。

## D. 前后差异

| 修改前 | 修改后 |
|---|---|
| 类别/整题关键词/直接consumer类型混合路由 | 不可变图hint、有向输入角色/数据流、集合优先 |
| 最终单体可能使上游提前缩小 | EXTREME/ORDINAL/RANK/SEM主体先保留DINO集合 |
| LOCATE整题prompt可能泄漏下游条件 | LOCATE只读TargetSpec；完整query只由SELECT_SEM消费 |
| 自由四数字、regex、重排/clamp、Region拒绝 | 严格JSON status、render像素、局部逆映射与原图界限验证 |
| scope代理框被当成功Entity | 无代理；NONE/AMBIGUOUS/UNRESOLVED明确0实体 |
| LOCATE Top-16、底层Top-K缺少传播 | 取消逻辑Top-16，worker/sidecar/tile截断状态向上传递 |
| 集合时跳过属性/CLIP相关性冒充真值 | 类别/属性/过滤分阶段trace；默认未决，可选有预算VLM三态核验 |
| 仅枚举关系/几何筛选 | 新完整字符串SINGLE/MULTI，NONE/歧义/未决/空集 |
| 视觉预算与池数量耦合 | 全池保存；默认8/批、最多评估64、总像素409600；超预算未决 |
| 新语义空集可能回整图猜答 | 新语义lineage阻止单体/最终猜答；COUNT可得0；旧SELECT兼容路径保留 |
| 旧REL缓存和受约束fallback | 保留模板/几何/评分；灰区超视觉预算显式未决 |

## E. 测试与未解决问题

分批回归 **617 passed，最终失败0**；另有Ruff和diff whitespace检查。完整命令、日志及初始失败修复见 [实施报告](locate_select_v2_implementation.md)、[验证清单](locate_select_v2_validation/manifest.json)。不是全仓所有测试，未运行真实模型。

无新增运行依赖，没有改竞赛HTTP/API、LEVIR-CC、最终序列化、Planner权重或训练数据。默认属性未校准时未决；示例配置启用VLM核验会增加调用。准确率、真实显存与延迟需独立实测。
