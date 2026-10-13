# TaskGraph DINO + GeoRSCLIP 接入记录

本次在 `team/sat-rs-vlm-georsclip-dino` 工作树中接入现有 CLIP 方法。遵照用户最后要求：不做真实图片或 GPU 测试；不使用 CLIP 未决判定阻止结果输出。任务文件中原先的 UNCERTAIN/ERROR 阻塞计数方案已被用户的最新指令替代。

## 调用链

| 入口 | 实际调用链 |
|---|---|
| LOCATE 的 DINO 分支 | LocateExecutor → VerifiedDetectionProvider → ProposalDetectionAdapter → StrategyProposalProvider → CountingSystem 既有切图/融合/实例复检 → LAE sidecar → GeoRSCLIPCandidateVerifier → 框集合 |
| COUNT(image/Region) | CountExecutor → VerifiedCountingProvider → CountingSystemProvider → 同一切图/融合/实例复检机制 → GeoRSCLIPCandidateVerifier → 筛后实例数 |
| COUNT(EntitySet) | 直接统计上游保留实例，不重复跑 DINO/CLIP。CLIP 的输出标记允许部分候选仍给出整数。 |
| DIRECT_DETECTION 简单计数 | 局部规范化 target 和 modifier，调用同一个 VerifiedCountingProvider，返回整数。 |

现有 VLM 单目标定位主路径保留。实际进入 DINO 的 LOCATE 分支关闭旧 CLIP top-K rerank，CLIP 属性判断只执行一条主路径。DINO 单目标回退有多个保留框时按综合分取第一项。未改 Planner、DSL、4B 共享模型、历史记忆或 Protocol 2.0 最终回答器。

## 已接入的评分规则

1. 大类初筛分差 = 固定正提示词集成分数 − 固定负提示词集成分数。
2. 限定词分 = `0.8 × (限定目标 − 无限定目标) + 0.2 × (限定目标 − 非该限定目标)`。
3. 综合排序分 = `0.5 × 大类分差 + 0.5 × 限定词分`。
4. 大类分差和所有必须满足的属性分差均 > 0 才保留；其他有分数的情况直接剔除，包括分差等于 0。多个属性用 AND 判定，排序使用各属性分差均值。
5. 没有限定词时只做大类初筛。

目标与属性全部从 `TargetSpec.category/attributes` 传入。例如 `car + color=blue` 生成 `blue car`、`car`、`non-blue car`。直接计数的 `red cars` 等表达局部拆解为目标词和属性；上游未提供结构化属性时可从简单计数问题的前置修饰词补齐。

大类提示词沿用 `configs/prompts/georsclip_major_categories.yaml`，正负名词固定，使用原先三种模板，不随机挑负词。车辆组仍为 7 个正词和 8 个负词，共 45 条。颜色、尺寸、状态等上游属性会套用同一混合评分方式，不用白名单阻止输出。`sample_checked_attributes` 只记录此前试过的颜色词，没有过滤开关作用。

CLIP 候选决策只有 MATCH/REJECT。CLIP 分数是图文相似度差，不是校准概率；实体的 score 继续保留 DINO 分数，综合分记录在 provenance 并用于排序。

## 输出与回退

- LOCATE 返回保留框的 EntitySet；无保留框返回空集合。
- COUNT 始终返回基于现有保留实例的整数，包括 0。
- CLIP 模型/权重不可用、文本组不匹配或评分失败时，保留可用 DINO 候选，记录 `DINO_fallback` 及错误原因，继续输出。
- DINO 命名策略中某 tile 失败时保留已完成 tile 的候选，融合后继续评分和计数。没有任何可用候选时输出空集合/0，trace 记录检测故障和回退状态。
- 必需 DINO/CLIP 配置或权重无法初始化时使用带诊断的候选/评分回退适配器，避免这两个 Provider 初始化失败阻断输出。
- 底层 count 缺失、与逐实例候选不一致、候选截断或实例复检诊断不完整，都保留原诊断，并以现有候选生成输出。

`candidate_pool_complete` 保留真实完整性诊断；新 CLIP 输出具有 `allow_partial_output=True` 和 `count_output_available=True`，这些诊断不会阻止新流程的 SELECT/COUNT/bbox 消费。旧的未启用验证流程保留既有严格检查。输出的 `count_status` 区分 FILTERED 和 AVAILABLE_CANDIDATES_FALLBACK，不产生 CLIP 未决状态。

保留逐候选原始 ID、全局 bbox、DINO 类别/分数、CLIP 模型标识、实际提示词、分项分数及判定原因。实例拆分子框使用原 ID 的派生 ID，并记录 source_candidate_id。无效坐标或重复 ID 直接剔除并记录原因。

## 检测策略与配置

生产配置已启用新方法：`configs/taskgraph/runtime.real.example.yaml`。现有 `configs/competition/protocol_v2.yaml` 引用该配置。

| 类别 | 策略 | 参数 |
|---|---|---|
| car、ship | dense_small | 800×800，overlap_ratio=0.15，DINO 阈值 0.2，关闭空间 resize，保留模型归一化及 padding。 |
| building、house | large_multi | 真正的 2×2 象限，overlap_ratio=0.15，DINO 阈值 0.2，模型默认 resize；没有额外 800 切图。 |
| 其他 | legacy | 沿用原配置。 |

两条入口复用 counting_system.tiling.plan_tiles 和既有融合规则。命名策略走 raw proposal 接口，无每 tile/全局固定 top-K 截断。融合后复用 postprocess_instances 和最多 4 次局部复检。

DINO 阈值与是否 resize 为请求级参数，因此 LOCATE/COUNT 继续共享现有 LAE worker。Region 使用实际像素裁剪偏移恢复原图坐标，输出保持 pixel_xyxy。

GeoRSCLIP 沿用原 bbox 裁剪、零额外上下文、灰色正方形补边；同一个验证器服务 LOCATE/COUNT，批量处理所有候选，图像嵌入与提示词原型均使用有界缓存。没有新增第二套 CLIP 模型加载器。

## 文件职责

| 文件 | 修改职责 |
|---|---|
| integrations/retrievers/candidate_verification.py | 可复用验证接口、二分决策、多属性 AND、DINO/CLIP 回退、检测/计数包装器。 |
| integrations/detectors/dino_georsclip.py | 原评分器增加既有 ProposalResult 输入和无属性粗筛入口。 |
| integrations/retrievers/openclip.py、protocol.py | 提示词组批量评分、完整 bbox 灰色补边、嵌入与文本原型缓存。 |
| integrations/detectors/strategies.py、integrations/counting/proposals.py | 统一策略选择、LOCATE 对现有 CountingSystem 切图和融合能力的复用。 |
| 04_counting_system_plan/src/counting_system/tiling.py、image_ops.py、executor.py | 800/2×2 规划、正确像素偏移、融合与复检、tile 失败保留已检出部分。 |
| integrations/counting/adapter.py、bridge.py、detector_bridge.py | 保留上游类别、全局坐标和 ID，发送 raw 请求，传播诊断。 |
| integrations/detectors/lae_dino_sidecar.py、scripts/integrations/lae_dino_worker.py | 请求级阈值/resize，复用批量共享 worker。 |
| taskgraph/providers.py、operators.py、runtime.py、executor.py、runtime_types.py | 正式入口接线、新 CLIP 输出策略允许部分结果消费、故障诊断。 |
| configs/taskgraph/runtime.real.example.yaml | 新验证策略启用，独立于旧 rerank；模型和缓存配置。 |
| tests/unit/taskgraph/test_georsclip_verification.py | Fake DINO/CLIP 的正式链路、输出保障和坐标/计数契约。 |

## 检查记录

本次只执行 Fake Provider 和合成图接口检查，没有加载真实 DINO/CLIP 或测试实际样本。195 项通过。可移植检查摘要：`reports/evaluation/georsclip_taskgraph_integration/summary.json`。测试命令生成的 `unit-tests.xml` 保留在本机。

```powershell
python -m pytest tests/unit/taskgraph/test_georsclip_verification.py tests/unit/taskgraph/test_counting_integration.py tests/unit/taskgraph/test_locate_routing_v2.py tests/unit/taskgraph/test_locate_select_v2.py tests/unit/taskgraph/test_runtime.py tests/unit/taskgraph/test_hardening.py tests/unit/taskgraph/test_regionset_contracts.py tests/unit/taskgraph/test_detection_phrase.py tests/unit/taskgraph/test_recall_fallback.py tests/unit/integrations/test_detector_providers.py tests/unit/integrations/test_detector_image_transport.py tests/unit/integrations/test_detector_parallelism.py --basetemp=.tmp/pytest-georsclip-always-output --junitxml=reports/evaluation/georsclip_taskgraph_integration/unit-tests.xml -q
```

修改文件的 Ruff 检查通过；`git diff --check` 通过；生产配置结构验证通过。真实图片/GPU 端到端测试按用户要求跳过，没有新增真实测试结果文件。此前示例是定性结果，未新增精确率、召回率、计数误差或概率校准结论。
