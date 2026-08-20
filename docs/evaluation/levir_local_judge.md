# LEVIR-CC本地小模型语义评审

## 目的

LEVIR-CC主模型继续自由生成变化描述，不再为了统计准确率进行第二次图像问答，也不要求主模型学习结构化0/1输出。评测完成后，本地文本小模型只读取`prediction`，判断这段描述表达的是：

- `0`：未表达建筑或永久结构的实质变化，包括只出现车辆、光照、阴影等临时或成像差异；
- `1`：表达了建筑、道路或其他永久结构的新建、拆除、扩建、移除或替换；
- `U`：描述矛盾、信息不足或无法确定，需要人工审计。

评审器不读取图片，也不能在作出判断时读取`metadata.changeflag`。`changeflag`只在评审结束后用于统计主模型变化检测性能。

## 运行流程

主模型推理配置默认关闭历史二次0/1推理：

```yaml
generation:
  change_binary_enabled: false
```

准备本地Qwen3-1.7B权重后，运行：

```powershell
python scripts/evaluation/judge_change_captions.py `
  --predictions E:\path\to\predictions.jsonl `
  --model E:\太空智算\models\Qwen3-1.7B `
  --output-dir E:\太空智算\evaluation-results-local\local-judge `
  --routing cascade `
  --batch-size 8
```

默认禁止联网加载模型。只有明确需要下载时才传入`--allow-download`。

随后使用v1.7契约评测：

```powershell
python scripts/evaluation/evaluate_predictions.py `
  --predictions E:\太空智算\evaluation-results-local\local-judge\judged_predictions.jsonl `
  --output-dir E:\太空智算\evaluation-results-local\local-judge-evaluated `
  --contract configs/eval/evaluation_contract_v1.7.yaml
```

v1.7严格模式会检查每条LEVIR-CC结果都具有本地评审器写入的
`prediction_changeflag=0/1`及`binary_prediction_source=local_*`。直接把主模型的原始
`predictions.jsonl`交给v1.7离线评测会停止并提示先运行本评审器，避免静默回退到旧的
Caption关键词规则、导致不同模型被粗粒度地判为相同结果。

主模型推理也必须为每个模型/检查点使用新的独立输出目录，例如：

```powershell
python scripts/evaluate_rs_vlm.py `
  --config configs/eval/qwen3vl_eval.yaml `
  --checkpoint E:\path\to\checkpoint `
  --output-dir E:\太空智算\evaluation-results-local\model-a-run-001
```

该目录会产生`model_run_manifest.json`，记录评测数据、配置、适配器/检查点指纹、生成设置和
预测文件SHA256。目录非空时推理会拒绝写入，防止后一次模型运行覆盖前一次预测。

## 输出

- `judged_predictions.jsonl`：保留原始记录，增加评审结论、来源、模型版本和耗时；
- `judge_summary.json`：覆盖率、待审计率、来源分布和评审耗时；
- `judge_manifest.json`：输入SHA256、模型、提示词版本和运行环境；
- `judge_audit_queue.jsonl`：所有`U`或无法解析的样本。

`binary_prediction_source`取值包括：

- `local_semantic_rule`：保守规则确认完整的无变化陈述；
- `local_semantic_positive_rule`：高置信规则确认道路、建筑等永久结构变化；
- `local_semantic_non_target_rule`：高置信规则确认仅含植被、外观等非目标差异；
- `local_llm_judge`：本地小模型给出0/1；
- `local_llm_judge_uncertain`：小模型返回U或输出非法，需要人工复核。

## 指标解释

必须分开报告两层结果：

1. 评审器语义有效性：与人工Caption语义标签比较的Accuracy、Macro F1、两类Recall、Coverage和Uncertain Rate；
2. 主模型任务能力：评审结论与图片级`changeflag`比较的Accuracy、F1、FNR、FPR、MCC和Kappa。

图片级`changeflag`不能代替人工Caption语义标签验证评审器。例如真实图片有变化，但主模型描述“场景无变化”时，评审器应忠实输出0；之后0与真实标签1不一致，才构成主模型错误。

## 人工语义盲审

正式采用小模型判定前，必须用人工标注 Caption 的含义。盲审不得查看图片、参考答案和
`metadata.changeflag`，否则会把“主模型是否看对图”和“文本判定器是否看懂 Caption”混为一谈。

从基础模型自由描述中分层抽取 300 条，并强制纳入新旧判定分歧：

```powershell
python scripts/evaluation/prepare_change_caption_audit.py `
  --predictions E:\path\to\base-predictions.jsonl `
  --comparison-judged E:\path\to\comparison-judged.jsonl `
  --local-judge-results E:\path\to\audit-judged.jsonl `
  --output-dir E:\path\to\human-audit `
  --sample-size 300
```

两位成员分别填写 `annotator_a.csv` 和 `annotator_b.csv`。分歧完成裁决后运行：

```powershell
python scripts/evaluation/evaluate_change_judge_audit.py `
  --annotator-a E:\path\to\annotator_a.csv `
  --annotator-b E:\path\to\annotator_b.csv `
  --answer-key E:\path\to\answer_key.json `
  --adjudicated E:\path\to\adjudicated.json `
  --output-dir E:\path\to\audit-summary
```

如果当前只有一位标注者，可暂时省略 `--annotator-b` 生成单人初步报告。输出会明确标记为
`single_annotator_preliminary`，不会计算一致率或 Cohen's Kappa，也不能作为最终人工金标准。

完整口径和示例见[变化描述语义盲审标注规范](change_caption_annotation_guide.md)。

## 三级离线级联与独立保留集

人工开发审计发现，小模型容易漏掉长Caption前部的道路/建筑变化，同时可能把植被和
外观差异误判为目标变化。v2.3混合实现按以下顺序处理：

1. 高置信整体无变化规则；
2. 高置信永久结构变化规则；
3. 高置信非目标差异规则；
4. 其余模糊文本才交给本地小模型。

前三步不产生额外模型调用。开发集结果只能用于调试，正式比较必须排除全部开发Caption：

```powershell
python scripts/evaluation/prepare_change_caption_audit.py `
  --predictions E:\path\to\base-predictions.jsonl `
  --exclude-captions E:\path\to\development-human-gold.csv `
  --output-dir E:\path\to\blind-holdout `
  --sample-size 200 `
  --seed 20260816
```

`audit_manifest.json`记录排除文件SHA256和排除Caption数量。保留集完成双人标注和裁决前，
不得查看隐藏答案键或根据保留集结果修改规则。

## v2.3当前状态

截至2026-08-16，双人独立标注、分歧裁决和独立保留集验证已经完成。当前冻结实现为
`levir-local-text-judge-v2.3-hybrid`，判定配置为`local_text_judge_priority_v1.3`。

191条二元人工Caption语义金标准结果（另外9条`U`不进入二分类分母）：

- Accuracy：86.39%；
- Balanced Accuracy：85.41%；
- Change Precision：83.33%；
- Change Recall：94.34%；
- Change F1：88.50%；
- TP=100、TN=65、FP=20、FN=6；
- 相对旧上下文解析器，Accuracy提高30.89个百分点。

1,333条LEVIR-CC回放模型结果：

- Accuracy：89.65%，Balanced Accuracy：89.65%；
- Change Precision：97.49%，Change Recall：81.41%，Change F1：88.73%；
- TP=543、TN=652、FP=14、FN=124。

v2.3专门区分两类容易混淆的表达：`buildings appear to be unchanged`中的`appear`
表示“看起来”，不能据此认定目标出现；`the appearance of a small structure`中的
`appearance of`则表示一个永久结构出现。该修补消除了v2.1/v2.2在人工样本上的定向回归，
且没有改变正式评测集总体结论。

14条假阳性预测均在文本中明确声称建筑、房屋、道路或路径出现/建成，因此文本语义判为
“有变化”符合本评审器目标。由于提交材料不包含对应双时相原图，目前尚不能进一步区分
模型幻觉和数据集标签问题；正式报告必须保留这一图像级审计限制。

## 历史字段兼容

人工验证CSV和JSON中的`hybrid_v2_decision`、`hybrid_v2`是已冻结的历史字段名。为保证旧结果、
脚本和绘图继续可读，v2.3不会重命名这些字段。判断实际算法版本必须读取：

- `implementation_version=levir-local-text-judge-v2.3-hybrid`；
- `decision_profile=local_text_judge_priority_v1.3`。

## 提交前验证记录

- v2.3已在191条冻结人工金标准及1,333条完整LEVIR-CC结果上复跑；
- 完整评测模块回归共74项、107个子测试通过；
- 提交前测试改用具有正常目录权限的隔离临时目录，此前Windows临时目录清理错误已排除；
- 评测脚本、评测模块和对应测试的Ruff静态检查通过；
- 模型权重、完整预测、人工标注原表和本地评测目录不进入Git提交。

更完整的验证口径和结论见[v2.3验证说明](levir_local_judge_validation_v2.3.md)。
