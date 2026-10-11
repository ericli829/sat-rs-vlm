# 程序化定位训练数据质量报告

实际生成 2000 条，划分 {'train': 1800, 'validation': 200}。一节点 LOCATE 共 1400 条，占 70.0%。
所有问题为英文。无图像、无真实答案框；仅监督规划路径。唯一性属于 synthetic planning assumption，Executor 必须拒绝零目标、多目标和未决证据。

## 静态检查

逐条使用当前 Schema、Validator、TypeChecker、compiler/parser 规范化往返及约束语法完整接受检查；独立校验结构化参数、自然语言、最短算子路径与生产输入。
每字符前缀在独立验证命令的 --every-prefix 模式检查；不能把完整接受检查单独称为已逐字符穷举。
~~~json
{
  "total": 2000,
  "split_counts": {
    "train": 1800,
    "validation": 200
  },
  "families": {
    "category": 900,
    "semantic": 100,
    "region": 300,
    "intrinsic": 500,
    "ordinal": 100,
    "extreme": 100
  },
  "categories": {
    "airplane": 57,
    "antenna": 66,
    "basketball court": 22,
    "bridge": 55,
    "building": 63,
    "bus": 58,
    "car": 53,
    "cargo ship": 66,
    "chimney": 70,
    "container": 62,
    "dock": 46,
    "fishing boat": 57,
    "football field": 55,
    "greenhouse": 55,
    "hangar": 23,
    "helipad": 67,
    "industrial building": 15,
    "oil tank": 67,
    "parking area": 59,
    "railway station": 62,
    "residential building": 24,
    "roundabout": 15,
    "runway": 18,
    "ship": 58,
    "silo": 61,
    "solar panel": 18,
    "sports field": 58,
    "stadium": 70,
    "storage tank": 60,
    "swimming pool": 66,
    "tennis court": 52,
    "tower": 45,
    "train": 56,
    "transmission tower": 15,
    "truck": 60,
    "vehicle": 63,
    "warehouse": 25,
    "water tower": 74,
    "wind turbine": 59,
    "yacht": 25
  },
  "attributes": {
    "color": 186,
    "has_part": 48,
    "shape": 144,
    "size": 91,
    "state": 32,
    "pattern": 8
  },
  "operators": {
    "LOCATE": 2000,
    "SELECT": 300,
    "REGION": 300
  },
  "select_modes": {
    "SEMANTIC": 100,
    "ORDINAL": 100,
    "EXTREME": 100
  },
  "node_counts": {
    "1": 1400,
    "2": 600
  },
  "final_types": {
    "BBOX": 2000
  },
  "one_node_locate_count": 1400,
  "one_node_locate_fraction": 0.7,
  "validation_pass_counts": {
    "schema": 2000,
    "validator": 2000,
    "type_checker": 2000,
    "canonical_roundtrip": 2000,
    "constraint_grammar": 2000,
    "production_input": 2000,
    "structured_semantics": 2000,
    "minimal_path": 2000,
    "bbox_static_source": 2000
  },
  "exact_text_duplicates": 0,
  "cross_split_exact_text": 0,
  "cross_split_categories": 0,
  "cross_split_template_families": 0,
  "cross_split_near_template_skeletons": 0,
  "within_split_reused_skeleton_rows": 1935,
  "dsl_characters_mean": 77.819,
  "dsl_characters_max": 168,
  "structural_split": "Graph skeletons repeat intentionally; category, action-template families, and available spatial rule values are held out. One-node LOCATE has no further structural variant to hold out."
}
~~~

## 划分与重复

按类别、动作句式家族以及可分离的空间规则值联合隔离。训练使用祈使句，验证使用独立疑问句，类别互斥。
同一 split 内有意复用模板（统计见 within_split_reused_skeleton_rows）。跨 split 广义语言模板骨架、类别及原文交叉均应为 0。
LOCATE→BBOX 等核心图骨架在训练和验证都出现，这是专项任务而非未见算子组合的泛化评测；不能声称结构完全独立。验证同时留出类别、句式和空间规则，不能单维归因；vehicle/car 等父子类别仍可能分属两侧。

## 输入与训练

每条 user 消息由生产 CompetitionRequestAdapter.adapt 和 Qwen3VLPlannerProvider._messages 直接构造：FREE_FORM、choices=null、符号 image0 引用，以及当前真实 bbox 定位后缀。
直接导出 system/user/assistant messages，assistant 仅规范 DSL。不经过 accepted 教师通道，不伪造 answer_audit。
512 输出 token 与 1088 序列预算、真实训练 Dataset/collator 的 assistant-only loss 检查记录在 sft_encoding_verification.json；没有该文件时不能宣称真实 tokenizer 已验证。

## 原数据保护和限制

现有 SFT、hard curriculum、固定 held-out test 和训练配置均不覆盖。旧 80 条问题只用于精确排重，不充当独立新评估集。
mix_planner_synthetic 支持独立构建默认 20% 增量曝光混合集，保留原训练行和固定 test 字节。未启动训练或真实模型推理。

## 本次全量实际验证

静态与规范化往返：2000/2000；每字符前缀157638次，19969个唯一前缀全部通过。
同配置完整重新生成后raw/train/validation三文件SHA256逐一相等。
真实本地tokenizer和训练collator：2000/2000，assistant-only mask精确，0截断，所有token ID合法。输出最大55 tokens（预算512），完整序列最大994（预算1088）。
五合同CPU/mock及相关回归合计614 passed、1 skipped（Windows symlink权限）、0 failed。
新增混合曝光20.0081%；固定test字节不变。旧train有118条当前DSL类型不合法，已独立记录并原样保留，不能宣称混合集全通过。
具体证据见static_verification.json、sft_token_integrity_verification.json、reproducibility_verification.json、validation_summary.json与implementation_report.md。
