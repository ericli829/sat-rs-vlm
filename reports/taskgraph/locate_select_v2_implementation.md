# LOCATE / SELECT v2 实施报告

2026-10-10。依据 `D:/Downloads/codex_locate_select_semantic_refactor_v2.md` 和用户本轮明确约束实施。分支 `codex/competition-protocol-v2`，开始时HEAD `7ee24b5`。完成实际代码与回归，未提交或推送。

## 完成结果与范围

Planner仍决定完整逻辑图；Executor仅对LOCATE预计算物理执行hint，没有插入、删除、替换LOCATE/SELECT/COUNT。集合主体走LAE-DINO，单体目标走现有Qwen3-VL。SELECT_SEM只筛选已有候选，完整字符串存于 `params.query`，不对背景树/草/道路额外检测。

取消LOCATE无条件Top-16，分离CPU完整检测池与VLM视觉/评估预算。新增候选完整性保护、固有属性核验策略、局部单体坐标和失败状态。旧SELECT_REL/RANK/ORD/EXTREME/SUBREGION保留；超预算关系灰区或底层已截断候选显式未决。

开始时已有competition API、hash memory、HF cache等用户改动，均保留。本轮未编辑competition目录、`runtime_memory.py`、`hf_vlm_engine.py`、训练JSONL或模型权重。`runtime.py`同时含原用户修改，不能把其相对HEAD的整份diff算作本轮贡献。无新增依赖，未改HTTP/API、LEVIR-CC和最终序列化。独立COUNT(image|Region)继续原Counting System。

## LOCATE路由决策

| 现有图用途 | requirement / 主后端 | 原因 |
|---|---|---|
| COUNT.entities | SET_REQUIRED / LAE_DINO | 完整实例基数 |
| SELECT_REL.candidates→COUNT | SET_REQUIRED / LAE_DINO | 筛后仍需枚举 |
| SELECT_EXTREME(TOPMOST)→ATTRIBUTE | SET_REQUIRED / LAE_DINO | 在全集比较极值，不能提前单体化 |
| SELECT_ORD(index=4) | SET_REQUIRED / LAE_DINO | 枚举和序数 |
| SELECT_RANK(bbox_area) | SET_REQUIRED / LAE_DINO | 比较面积 |
| GROUP.entities | SET_REQUIRED / LAE_DINO | 保留分组集合 |
| SELECT_SEM.candidates | SET_REQUIRED / LAE_DINO | query留给SELECT消费 |
| 仅SELECT_REL.reference，另一路被计数 | SINGLE_TARGET / VLM_SINGLE | reference不传递另一候选流的COUNT需求 |
| white square building→ATTRIBUTE | SINGLE_TARGET / VLM_SINGLE | 当前TargetSpec指定单体，无集合消费者 |
| LOCATE直接final定位 | SINGLE_TARGET / VLM_SINGLE | 无集合比较路径 |
| 共享给集合和单体消费者 | SET_REQUIRED / LAE_DINO | 集合优先，一份一致输出 |
| EntitySet→SUBREGION、未知evidence cardinality | SET_REQUIRED / LAE_DINO | ambiguous_set_requirement |
| 真Region→SUBREGION→COUNT.image的兄弟链 | 不污染该LOCATE | Region切片不是枚举建筑 |

只分析有向inputs引用、role、op及SELECT.mode，跟踪集合流经SELECT/GROUP；reference边终止单体用途传播。不读取原问题、类别、图像尺寸、数据集。hint为frozen dataclass，不写入TaskGraph持久化JSON。

第5完整图的实际trace摘录；[完整记录](locate_select_v2_after/e2e_05.json)：

```json
[
  {"node":"n1", "requirement":"SINGLE_TARGET", "route_reason":"single_consumer",
   "consumer_paths":["n1 -> n3:SELECT.reference"],
   "planned_backend":"VLM_SINGLE", "actual_backend":"VLM_SINGLE", "fallback_reason":null},
  {"node":"n2", "requirement":"SET_REQUIRED", "route_reason":"set_consumer",
   "consumer_paths":["n2 -> n3:SELECT.candidates", "n2 -> n3:SELECT.candidates -> n4:COUNT.entities"],
   "planned_backend":"LAE_DINO", "actual_backend":"LAE_DINO", "fallback_reason":null,
   "category_candidate_count":20, "final_candidate_count":20}
]
```

仅VLM UNRESOLVED按 `single_detector_fallback` 回退DINO；NONE/AMBIGUOUS不造框。多候选回退有预算地确认身份，不能取检测最高分冒充唯一目标。TargetSpec之外的整题和下游query均不进入LOCATE。fallback_reason、属性阶段、最终状态可追踪。

## SELECT_SEM及端到端执行

第3图由当前compiler生成并经parser/Validator检查：

```text
INTENT(OTHER)
n1=LOCATE($image0,T("building"))
n2=SELECT_SEM($n1,null,"the building with trees to its right and grass below",SINGLE)
FINAL_QUESTION($n2,CHOICE_SINGLE,"Where is the selected building?")
```

对应n2：

```json
{"id":"n2","op":"SELECT","inputs":{"candidates":"$n1"},
 "params":{"mode":"SEMANTIC","query":"the building with trees to its right and grass below",
           "selection_type":"SINGLE"}}
```

实际执行mock结果：DINO3个框→VLM有限选择B→`candidate_0002`，复用原Entity及原图框`[180,160,250,220]`，status=OK，没有新检测。完整query只出现在SELECT_SEM。见 [e2e_03](locate_select_v2_after/e2e_03.json)。MULTI例：`SELECT_SEM($n1,null,"red cars near the road",MULTI)`。JSON可显式scope；四参DSL无法无损编码额外scope时compiler明确报错。

| 决策/证据 | SINGLE | MULTI |
|---|---|---|
| 唯一正项/正常正项集合 | OK，一个既有候选 | OK，既有候选子集 |
| 全NONE/NO_MATCH，或输入为空 | EMPTY | EMPTY，合法空集 |
| 多个正项/AMBIGUOUS | AMBIGUOUS | 多项合法 |
| UNRESOLVED/UNCERTAIN、非法输出、超预算 | UNRESOLVED | UNRESOLVED，已确认正项仅留作审计 |

SINGLE每批有限候选为A..H/NONE/AMBIGUOUS/UNRESOLVED，跨批多个匹配不能取第一个。MULTI逐候选做真正MATCH/NO_MATCH/UNCERTAIN三态cached scoring，不确定不当否。无自由文本ID或新bbox解析回退。无缓存能力、非有限分数、ID/映射错、selected ID与最高分不一致、终端平分均未决。

新语义空集可供COUNT得0；不能供ATTRIBUTE等单体消费者或最终整图猜答。嵌套SELECT传播语义来源，避免绕过该边界。旧SELECT既有兼容空集通路保留。

五个合法完整图均实际执行GraphExecutor，前后逻辑JSON相等：

| 链路 | mock结果 | 证据 |
|---|---|---|
| 白色方形建筑→ATTRIBUTE | VLM bbox `[100,60,200,140]`→white | [01](locate_select_v2_after/e2e_01.json) |
| building→TOPMOST→ATTRIBUTE | 20候选中低分第20项y=5被选→white | [02](locate_select_v2_after/e2e_02.json) |
| building→SEM_SINGLE(树右草下) | candidate_0002，OK | [03](locate_select_v2_after/e2e_03.json) |
| red cars→SEM_MULTI→COUNT | 类别3→属性2→语义1→COUNT=1 | [04](locate_select_v2_after/e2e_04.json) |
| car→REL→COUNT，building为reference | car DINO20，reference VLM1；LEFT_OF10→COUNT=10 | [05](locate_select_v2_after/e2e_05.json) |

第3图保留合法CHOICE_SINGLE残余问题；验证止于final source，不评测benchmark最终选项。所有预测均mock，不能解读为真实模型准确率。

## 候选、坐标和资源

- 已返回检测候选完整保存在CPU逻辑池；旧max_candidates=16兼容接收但不截断。COUNT.entities不重检测。阈值/NMS下的检测池完整不等于真实世界100%召回。
- worker记录NMS后Top-K前数量和精确截断；client/tile/adapter向上传递不完整状态。旧worker无标记且达到Top-K、global_top_k饱和时保守未决。阈值/NMS/模型结果不变。部署需同步worker文件并由用户重新加载旧常驻sidecar；本轮未启停服务。
- 默认8候选/批、最多评估64、每次视觉总像素409600。>64逻辑池仍保存，未见ID明确列出并标未决。REL仅语义灰区受8预算，确定性关系处理全池。
- 推理串行，复用既有2B；无新增模型或GPU并行流。沿用推理→有限决策KV cache及finally释放。SINGLE最多8次语义决策，MULTI/属性核验最多64次；不同候选仍重新prefill，有时间代价。
- InputComposer使用候选/参考标记画布和scope overview；32像素stride、每视觉至少65536像素，还受provider.max_pixels限制。无法满足预算/比例显式未决。未做实际processor前向或真实显存实测。
- 局部Region记录crop_origin_global/source_crop_size/render_size/scale_xy；严格JSON数值、顺序、范围、面积后逆映射，再检查scope和原图边界。拒绝regex猜框、NaN、反序、超界与scope代理。
- DINO类别生成、属性核验、最终过滤分别trace。attribute_policy默认unresolved；未验证属性保留审计但不能精确消费。示例显式vlm时三态核验，亦限64候选。未校准CLIP相关性不当作属性真值。

## 修改文件及原因

只列本轮改动；开始时已有文件改动不在此表。快照里的未修改源码仅供审计。

| 文件 | 原因 |
|---|---|
| `src/sat_rs_vlm/taskgraph/locate_routing.py`（新） | 独立不可变职责分析、路由trace |
| `src/sat_rs_vlm/taskgraph/execution_plan.py` | LOCATE hint接入预计算 |
| `src/sat_rs_vlm/taskgraph/operators.py` | LOCATE路由/完整池/属性；新SEM；COUNT完整性和REL预算 |
| `src/sat_rs_vlm/taskgraph/semantic_selection.py`（新） | 有预算字符串筛选、三态验证、ID恢复 |
| `src/sat_rs_vlm/taskgraph/referent_refinement.py` | TargetSpec-only提示、严格JSON、局部坐标和失败配置 |
| `src/sat_rs_vlm/taskgraph/input_composer.py` | 可逆LOCATE视觉、候选及上下文总预算 |
| `src/sat_rs_vlm/taskgraph/providers.py` | 检测偏移/有效框/完整性、Entity/Region统一ID，复用原Qwen后端 |
| `src/sat_rs_vlm/taskgraph/executor.py` | 不完整池/语义空集消费边界 |
| `src/sat_rs_vlm/taskgraph/runtime.py` | 配置接线、最终证据保护，保留原memory改动 |
| `src/sat_rs_vlm/taskgraph/runtime_types.py` | CandidateIntegrityError防旧fallback吞未决 |
| `src/sat_rs_vlm/taskgraph/schema.py` | SEMANTIC、query、cardinality及字段约束 |
| `scripts/integrations/lae_dino_worker.py` | Top-K前数量和截断元数据 |
| `src/sat_rs_vlm/integrations/detectors/lae_dino_sidecar.py` | worker/client截断传播，旧worker饱和保守状态 |
| `src/sat_rs_vlm/integrations/detectors/tiled.py` | tile完整性和global_top_k饱和传播 |
| `taskgraph_lab/taskgraph/enums.py`、`schema.py` | Lab新模式/字符串/cardinality同步 |
| `taskgraph_lab/taskgraph/type_checker.py` | SINGLE/MULTI下游类型及scope角色 |
| `taskgraph_lab/taskgraph/dsl/compiler.py` | SELECT_SEM生成，不可无损scope报错 |
| `taskgraph_lab/taskgraph/dsl/parser.py` | 字符串进params.query，非inputs引用 |
| `taskgraph_lab/taskgraph/dsl/constraint.py` | JSON字符串与前缀约束语法 |
| `taskgraph_lab/prompts/planner_student_system_prompt.txt`、`system_prompt.txt` | 新SELECT DSL和职责说明 |
| `configs/taskgraph/runtime.real.example.yaml` | 单体/属性/语义预算配置及过期注释 |
| `docs/taskgraph_runtime.md`、`taskgraph_lab/specs/taskgraph_v1_1_dsl.md` | 使用和DSL文档 |
| `tests/unit/taskgraph/test_locate_routing_v2.py`（新） | 独立角色/共享/间接/Region/未知用途路由 |
| `tests/unit/taskgraph/test_locate_select_v2.py`（新） | 必验链、>16、三态、空集、局部框、预算、非法输出、完整e2e |
| `taskgraph_lab/tests/test_select_semantic_v2.py`（新） | Unicode/转义/前缀/类型/往返 |
| `tests/unit/taskgraph/test_locate_refinement.py`、`test_vlm_referent.py`、`test_recall_fallback.py` | 旧regex/scope代理/grid断言更新为新契约 |
| `tests/unit/taskgraph/test_select_system_hardening.py`、`test_v1_closure_e2e.py` | REL灰区预算和集合DINO路由断言 |
| `tests/unit/integrations/test_detector_providers.py` | Top-K状态和正常完整状态验证 |
| `scripts/taskgraph/audit_locate_select_prompts.py`（新） | 实际prompt捕获、immutable before、after与5合法图 |
| `reports/taskgraph/locate_select_v2_*audit.md`、本报告（新） | 前/后审计、差异、交付 |
| `reports/taskgraph/locate_select_v2_before/`、`locate_select_v2_after/`（新） | 源码哈希、render、chat template、图/DSL/trace/视觉 |
| `reports/taskgraph/locate_select_v2_validation/`（新） | 完整命令清单、最终日志 |

Validator既有validate_candidate入口使用更新Schema/type checker，无需另改validator.py。旧公开refine/grid入口保留，新LOCATE不再调用；可达性及全部保留VLM模板单独审计。

## 提示词审计

分别保存 [修改前](locate_select_v2_before_audit.md)、[修改后](locate_select_v2_after_audit.md)、[前后对照](locate_select_v2_before_after_audit.md)。原快照在执行代码修改前建立并禁止覆盖。

流程不足：最终choice间接回退在首次编码前未单列，之后依据未修改的choice.py补记；已明确披露，不能声称首次审计毫无遗漏。修改后捕获该入口、旧整图choice包装、REL受约束fallback、默认关闭的legacy生成fallback。

修改后14场景/19捕获调用含完整业务prompt、公共包装、实际chat template、options、推理指令、终端/允许值及suffix。无新增VLM系统包装；query仅SELECT消费。没有加载模型，reasoning正文mock占位，未生成processor展开后的视觉token。

## 测试和失败项

解释器全部为用户指定路径；命令形如：

```powershell
& 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe' -m pytest -q --basetemp=.tmp/新的唯一目录 <测试文件列表>
& 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe' -m scripts.taskgraph.audit_locate_select_prompts after
```

每批完整可复制命令及文件列表见 [manifest.json](locate_select_v2_validation/manifest.json)。每批<30秒，未启动长时后台任务。合计 **617 passed，最终失败0**：

| 批次 | 结果 | 日志 |
|---|---|---|
| 路由/新LOCATE+SEM/旧SELECT/Runtime/闭环 | 135 passed | [core.log](locate_select_v2_validation/core.log) |
| Lab Schema/Validator/type/DSL/constraint/prompts/adapters | 228 passed | [lab.log](locate_select_v2_validation/lab.log) |
| Provider/COUNT/composer/fusion/RegionSet/choice | 103 passed | [providers.log](locate_select_v2_validation/providers.log) |
| 现有HF cache/engine和Planner provider | 41 passed | [engine.log](locate_select_v2_validation/engine.log) |
| 其余TaskGraph含memory/路由 | 58 passed | [remaining.log](locate_select_v2_validation/remaining.log) |
| sidecar/tile/传输/CPU预取/CLIP集成单测 | 52 passed | [producers.log](locate_select_v2_validation/producers.log) |

另有Ruff与diff whitespace检查，验证目录保存结果。不是全仓所有测试。没有删除或xfail原回归。修复过的初始失败：旧假scope成功/自由四数字契约断言、新fixture必填target/answer_type、审计直接脚本缺lab导入（用模块命令）、fallback模拟条件；一次测试文件名输入错误后已按实际文件名重跑。pytest默认临时目录权限通过唯一basetemp绕开，未删用户目录。最终没有未解决测试失败。

## 未解决问题与真实实验

未运行真实Qwen/DINO权重前向、原图标注bbox准确率、processor视觉token检查、GPU显存峰值、端到端延迟实验；不是因为宣称设备/权重缺失。617项及五图均为单元/mock证据，不代表准确率提升。

需进一步实测：实际Qwen像素定位/局部逆映射误差、重复建筑与模糊条件的NONE/歧义可靠性、小目标在总409600像素下可辨性、多批SINGLE一致性、三态属性校准、DINO召回及阈值/NMS。MULTI/属性核验最多64次prefill可能增加时延；全CPU候选和详细trace体积会增长。输入预算不能保证已有planner+sidecar常驻负载下绝无OOM。

无实质性需求冲突。明确边界：JSON scope合法但四参DSL不能表达时拒绝编译；超预算未决；无可靠属性核验默认未决；旧公开refiner仅兼容保留。没有扩大到训练、竞赛协议、计数算法或新部署方案。
