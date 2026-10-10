# SELECT_SEM 可选 scope 的 DSL 表达补全

日期：2026-10-10。工作分支：`codex/competition-protocol-v2`。开始时 HEAD 为 `8ee1c788ba9c4ebcb17d943bd956f9a455dfc209`。

## 结论

已实现 `SELECT_SEM(candidates, reference, query, selection_type[, scope])`。现有四参数形式保持兼容，第五参数进入 `inputs.scope`。对于当前规范化 PlannerTarget 的 SELECT(mode=SEMANTIC)，没有发现仍缺少 DSL 表达的合法输入角色或参数字段。

最终相关测试共 **398 passed，0 failed**；Ruff 和本轮相关 tracked 文件的 `git diff --check` 通过。没有进行模型推理、训练或准确率实验。

修改保留在工作树，尚未提交或推送。本轮开始时已有 HTTP/API、HF 引擎、哈希记忆、DINO worker 等修改；实施期间也观察到其他性能文件继续更新。本轮只对下列指定位置作增量修改，保留其他工作树内容。尤其 `docs/taskgraph_runtime.md` 中其他批处理/性能段落的变化不属于本轮。

## 方案核对与实现边界

开始修改前核对了工作树、compiler、parser、约束解码语法、生产与实验室 Schema、实验室类型检查器及 Validator。旧实现的实际缺口：

1. compiler 在 SEMANTIC inputs 包含 scope 时显式抛出 DSLCompileError。
2. parser 将 SELECT_SEM 限定为四参数。
3. prefix grammar 没有第五参数分支。
4. 两处 Planner 契约说明和两份规范文档还描述旧限制。

已有类型检查器的 SELECT 签名为：

- candidates：EntitySet | Region | RegionSet，必填。
- reference：Entity | EntitySet | Region，可选。
- scope：ImageRef | Region，可选。

`OPTIONAL_INPUTS` 已列出 reference 和 scope；GraphNode 已拒绝 SELECT 的引用列表；Validator 已遍历所有 inputs 检查节点依赖、前序引用、环、未知输入及类型。因此 **Schema、type_checker.py、Validator 均无需修改**。新约束语法直接复用这些输入类型签名，没有另建一套类型规则，也没有改变已有联合输出类型的推断策略。

没有修改 LOCATE 路由/后端、SELECT_SEM 执行算法、运行时 VLM 提示词、64/8 推理预算、CLIP 策略、Planner 权重/架构/训练数据、竞赛接口或最终答案协议。更新的是 Planner 的序列化契约说明。

## 精确 DSL 定义

```text
SELECT_SEM(candidates, reference, query, selection_type[, scope])
```

| 位置 | 允许形式 | JSON 目标 |
| --- | --- | --- |
| 1 candidates | 已有候选引用，满足现有候选类型约束 | inputs.candidates |
| 2 reference | 合法参考物引用，或字面量 null | inputs.reference；null 表示缺省 |
| 3 query | 完整 JSON 字符串，使用既有非空/2048 字符限制 | params.query |
| 4 selection_type | 未加引号的 SINGLE 或 MULTI | params.selection_type |
| 5 scope（可选） | 合法 ImageRef 或前序 Region 引用 | inputs.scope |

第五参数不是 null 占位符，也不是字符串、列表或命名参数。省略第五参数时不插入默认 scope。scope 与 reference 可同时存在，二者使用不同类型签名；例如 ImageRef 可以作 scope，不能作 reference。其他 SELECT 模式的参数个数不变。

例：

```text
n1=REGION($image0,CENTER)
n2=LOCATE($n1,T("building"))
n3=SELECT_SEM($n2,$n1,"the building near the road",SINGLE,$image0)
FINAL_QUESTION($n3,CHOICE_SINGLE,"Which building matches?")
```

其中 n3 的 JSON 为：

```json
{
  "id": "n3",
  "op": "SELECT",
  "inputs": {
    "candidates": "$n2",
    "reference": "$n1",
    "scope": "$image0"
  },
  "params": {
    "mode": "SEMANTIC",
    "query": "the building near the road",
    "selection_type": "SINGLE"
  }
}
```

四参数的 `SELECT_SEM($n2,null,"the building near the road",SINGLE)` 不产生 reference 或 scope。五参数的 `SELECT_SEM($n2,null,"the building near the road",MULTI,$n1)` 只产生 scope，不产生 reference。

## 修改文件与原因

| 文件 | 本轮修改 |
| --- | --- |
| taskgraph_lab/taskgraph/dsl/compiler.py | 按字段是否存在追加第五参数，替换旧的 scope 拒绝分支；没有默认值注入 |
| taskgraph_lab/taskgraph/dsl/parser.py | 只允许 SELECT_SEM 使用 4..5 参数；第五个引用进入 inputs.scope，query/cardinality 的位置不变 |
| taskgraph_lab/taskgraph/dsl/constraint.py | 支持可选第五参数；SELECT_SEM 三种输入引用分别按现有类型签名筛选，并限制到调用方图像/前序节点 |
| taskgraph_lab/tests/test_select_semantic_v2.py | 将旧的“scope 应拒绝编译”断言改为 reference+scope 都保留的往返断言 |
| taskgraph_lab/tests/test_select_semantic_scope.py | 新增 135 个参数化测试用例，覆盖往返、作用域、字符串、类型、拓扑、前缀及导出链路 |
| taskgraph_lab/prompts/planner_student_system_prompt.txt | 告知 Planner 可选第五参数、独立角色和缺省语义 |
| taskgraph_lab/prompts/system_prompt.txt | 删除旧的不支持 scope 的陈述，保持 Teacher JSON 输出职责与其他规则 |
| taskgraph_lab/specs/taskgraph_v1_1_dsl.md | 补精确签名、映射、约束和四/五参数示例 |
| docs/taskgraph_runtime.md | 仅更正 SELECT_SEM 的 DSL scope 支持说明 |
| reports/taskgraph/select_sem_scope_20261010/implementation.md | 本实施和验收记录 |

## 测试与结果

所有命令在仓库根目录执行，解释器固定为：

```powershell
$testPython = 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe'
```

1. 修改前基线：

```powershell
& $testPython -m pytest -q --basetemp=.tmp/select-sem-scope-baseline taskgraph_lab/tests/test_select_semantic_v2.py taskgraph_lab/tests/test_type_checker.py
```

结果：14 passed in 1.99s。该基线仍包含旧“scope 编译报错”的断言，不计入最终 398 项。

2. 最终新增能力及原语义筛选测试：

```powershell
& $testPython -m pytest -q --basetemp=.tmp/select-sem-scope-focused-final taskgraph_lab/tests/test_select_semantic_scope.py taskgraph_lab/tests/test_select_semantic_v2.py
```

结果：**144 passed in 4.20s**。

覆盖：四参数缺省 scope；五参数 ImageRef/Region scope；reference+scope 同时存在；SINGLE/MULTI；EntitySet/Region/RegionSet 候选；SUBREGION scope；第二张输入图像与调用方输入校验；canonical 节点重命名保留 scope/reference 而不改 query；JSON→DSL→JSON；生产 GraphNode 接受解析结果；SFT 导出保留 scope。SFT 导出测试只写 pytest 临时 fixture，不改真实训练数据。

字符串覆盖：中英文、引号、反斜杠、字面量 `$n99`、类似 DSL 的字符串内容、换行/制表/回车/退格/换页、`\/`、Unicode escape 与代理对、2048 字符边界。对四/五参数与 SINGLE/MULTI 的转义程序逐字符检查前缀合法性。

拒绝覆盖：缺参数/多参数/尾逗号/命名参数/错位；null/字符串/列表 scope；非法 cardinality；无引号或非字符串 query；坏 JSON escape、未闭合字符串、未转义换行；空白和过长 query；原始 ImageRef 作 candidates/reference；EntitySet/整数/标签/布尔作 scope；scope 自引用、未来引用、未知节点；调用方缺少第二图像；其他 SELECT 模式擅自增加参数。

3. 既有 DSL 回归：

```powershell
& $testPython -m pytest -q --basetemp=.tmp/select-sem-scope-dsl-regression taskgraph_lab/tests/test_dsl_roundtrip.py taskgraph_lab/tests/test_dsl_invalid.py taskgraph_lab/tests/test_dsl_constraint.py taskgraph_lab/tests/test_dsl_integration.py
```

结果：**179 passed in 3.24s**。覆盖既有算子、SELECT_REL/RANK/ORD/EXTREME/SUBREGION、编译稳定性、逐字符约束语法、生成与导出集成。

4. Schema/Validator/类型/Planner 接入与提示词契约：

```powershell
& $testPython -m pytest -q --basetemp=.tmp/select-sem-scope-contract-regression taskgraph_lab/tests/test_type_checker.py taskgraph_lab/tests/test_schema.py taskgraph_lab/tests/test_validator.py taskgraph_lab/tests/test_prompt_operator_contract.py taskgraph_lab/tests/test_batch_parser_valid.py taskgraph_lab/tests/test_batch_parser_invalid.py tests/unit/taskgraph/test_prompt_contracts.py tests/unit/taskgraph/test_planner_provider.py
```

结果：**35 passed in 0.88s**。

5. 生产 SELECT 行为回归：

```powershell
& $testPython -m pytest -q --basetemp=.tmp/select-sem-scope-select-regression tests/unit/taskgraph/test_select_system_v1.py tests/unit/taskgraph/test_select_system_hardening.py
```

结果：**40 passed in 6.77s**。

6. 静态检查：

```powershell
& $testPython -m ruff check taskgraph_lab/taskgraph/dsl/compiler.py taskgraph_lab/taskgraph/dsl/parser.py taskgraph_lab/taskgraph/dsl/constraint.py taskgraph_lab/tests/test_select_semantic_v2.py taskgraph_lab/tests/test_select_semantic_scope.py
```

结果：All checks passed。一次新增测试行长问题已修正，最终无 lint 失败。相关 tracked 文件的 `git diff --check` 也通过。

最终四组测试互不重复，合计 **144 + 179 + 35 + 40 = 398**。较早的基线及开发中重复测试没有累加。所有测试命令都在本机 30 秒内完成，没有启动长时间后台任务。

## 表达完整性的边界

当前 SEMANTIC 模式合法的业务参数只有 query 和 selection_type；其合法输入角色为 candidates、可选 reference、可选 scope。这五个位置现在都有确定、可逆的 DSL 映射。没有添加新的 JSON 字段或语义参数。

“无损往返”按现有 `canonicalize_target` 定义比较规范化 PlannerTarget：既有节点拓扑规范化/重命名、枚举规范化、query 首尾空白去除和 JSON escape 写法规范化保持原行为。不会把这些既有规范化等同于原始文本逐字节一致。完整生产 TaskGraph 的 question/inputs 等外层请求信息仍由现有传输层提供，不是 Planner DSL 的新增参数。

引用使用既有规范的 `$imageN` / 前序 `$nN`。DSL 中未来、自身或不存在的节点引用被拒绝；调用方输入存在性在约束语法和带 inputs 的 Validator 中校验。compiler 继续沿用既有 canonicalization 流程，不改变 JSON DAG 拓扑排序策略。

prefix grammar 负责受约束的语法、引用范围及静态输入类型；query 的非空/长度和全图语义校验继续由解析后的 Schema/Validator 承担。没有将前缀可接受等同于整张 TaskGraph 已完成语义验证，也没有改变既有联合输出类型近似或运行时校验。

因此，对上述现有规范化合同，未发现仍然 JSON 可表达、SELECT_SEM DSL 不可表达的合法字段组合。此结论不是对其他 SELECT 模式或任意生产扩展字段的新增表达承诺。

## 后续问题集

接受“只需要问题、不需要参考图像”的方向。后续问题集可以从 VRS 文本抽取，也可以程序生成简单定位、复合条件、极值/序数和条件计数问句，无需为问句生成附带图片或答案。进入正式 Planner SFT 时再配套审核后的 DSL/TaskGraph 标签；模型执行和真实视觉准确率验证另行进行。本轮没有生成或改写问题集与训练数据。
