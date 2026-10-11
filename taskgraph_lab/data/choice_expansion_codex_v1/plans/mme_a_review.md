# MME A 分片中文审核记录

当前分片共 425 条真实英文选择题，已逐条阅读问题并建立对应的类别、固有属性、区域和选择规则；输出 425 条，待处理 0 条。生成器只应用这些经审阅的逐行规则，不以关键词猜测未审阅题目的 gold。没有读取答案索引、GT、图片或调用外部模型 API。

源文件 SHA256 为 163b3e08838ec8bf34c1a65135566fa767a19ec53b5cfbb3f8c2c222adee60a7，脚本在生成前强制核对字节哈希，源题顺序或内容改变即拒绝沿用旧标注。

## 检查结果

全部输出逐条通过 DSL 解析、带原问题和 inputs 的 validate_candidate，以及规范化 JSON → DSL → JSON 往返。Final 类型为 {'CHOICE_SINGLE': 423, 'CHOICE_MULTI': 2}；COUNT 节点 54，SELECT_SEM 节点 184，GROUP 节点 3。未添加 BBOX、VLM_REASON 或预测答案坐标。

## 本次二次语义复核

- 修正 37 个参考物位置被错误提升为目标整图裁剪的情况。参考物角落、外部邻接和相对参考区域方向保留在完整 SELECT_SEM query 中；只有明确属于目标自身的图像区域继续使用 REGION。
- 第 21 题保留“较大停泊区域的右下角”，没有将其改写为整图右下角；第 139 题保留水域自身的最左侧；第 294 题没有将“位于中心参考物上方”的建筑先裁进 CENTER。
- 明确的最大/最小面积、最左/最右/最上仍使用 RANK/EXTREME。第 2 题定位 roof 候选并按屋顶 bbox_area 排序，没有把建筑面积冒充屋顶面积。top car 先筛道路条件再执行 TOPMOST。
- 第 160 题恢复 GROUP ROW，第 177 题恢复 GROUP CLUSTER；第 353 题按“从左第二排码头”使用 COLUMN 分组、LEFT_TO_RIGHT 序数、子区域和 COUNT，避免由 SEM 代替明确的分组或顺序。
- 第 375、396、420 题的 tallest 不等于 bbox_area；当前 RANK 没有 height 标准，因此完整保留于 SELECT_SEM。第 388 题 first car 缺少排序方向，保留完整语义条件并标记歧义，没有编造 TOP_TO_BOTTOM。
- 第 83 题保留 fill，第 316 题保留 ground/background。白色航迹属于周边关联条件，第 341 题已从固有 has_part 中移出并交由 SELECT_SEM 消费。
- 多目标颜色使用视觉集合和静态原问题 FINAL_QUESTION，未将 MULTI 选择直接交给 ATTRIBUTE。单个建筑复合体的 exterior walls 可以作为 ATTRIBUTE 的 part，避免仅因表层 are 而错误判成多个目标。

## 限制

结构合法不代表真实图片上答案正确；本次没有执行视觉推理或模型准确率实验。源题含翻译错误或歧义的第 144、155、220、388、405 题已写入逐行 review_notes，执行时可能得到未决结果。此审核只覆盖 mme_a 分片，不修改其他分片、旧训练集或固定测试集。
