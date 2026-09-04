# 遥感指令数据格式

## 内部 `rs_*.jsonl`

每行是一个遥感指令样本：

```json
{
  "id": "sample_000001",
  "task_type": "captioning",
  "images": ["data/samples/demo_image.png"],
  "instruction": "请描述这张遥感图像中的主要地物。",
  "answer": "图像中包含建筑物、道路和植被区域。",
  "metadata": {"dataset": "sample", "split": "train"}
}
```

字段说明：

- `id`：全局唯一样本 ID。
- `task_type`：支持 `detection`、`counting`、`captioning`、`vqa`、`scene_classification`、`segmentation`、`change_detection` 和 `unknown`。
- `images`：一个或多个图像路径；变化检测使用两张图。
- `instruction`：用户指令。
- `answer`：监督微调目标回答。
- `metadata`：数据集名称、split、传感器、区域、分辨率等扩展信息。

## Qwen3-VL `messages`

单图格式：

```json
{
  "id": "sample_000001",
  "messages": [
    {"role": "user", "content": [{"type": "image", "image": "data/samples/demo_image.png"}, {"type": "text", "text": "请描述这张遥感图像中的主要地物。"}]},
    {"role": "assistant", "content": "图像中包含建筑物、道路和植被区域。"}
  ],
  "task_type": "captioning",
  "metadata": {"dataset": "sample"}
}
```

双图变化检测格式：

```json
{
  "id": "change_000001",
  "messages": [
    {"role": "user", "content": [{"type": "image", "image": "data/samples/before.png"}, {"type": "image", "image": "data/samples/after.png"}, {"type": "text", "text": "第一张为变化前，第二张为变化后。请描述两张遥感图像之间的变化。"}]},
    {"role": "assistant", "content": "变化后图像中新增了建筑物，道路区域基本保持不变。"}
  ],
  "task_type": "change_detection",
  "metadata": {"dataset": "sample_change"}
}
```

## 任务示例

detection：

```json
{"task_type":"detection","instruction":"Locate the aircraft. Return ONLY normalized_0_1 JSON.","answer":"{\"label\":\"aircraft\",\"bbox\":[0.1,0.2,0.3,0.4]}"}
```

counting：

```json
{"task_type":"counting","instruction":"请统计图像中的建筑物数量，只返回 JSON。","answer":"{\"count\":5}"}
```

captioning：

```json
{"task_type":"captioning","instruction":"请描述这张遥感图像中的主要地物。","answer":"图像中包含建筑物、道路和植被区域。"}
```

change_detection：

```json
{"task_type":"change_detection","instruction":"第一张为变化前，第二张为变化后。请描述变化。","answer":"变化后新增了建筑物。"}
```

## metadata 规范

建议字段包括：

- `dataset`：来源数据集名称。
- `split`：train/val/test。
- `sensor`：传感器名称。
- `resolution`：空间分辨率。
- `region`：区域或城市。
- `license`：数据授权信息。

MME-RealWorld-RS 和 XLRS 的官方评测样本还应保留
`dataset_version`、`split`、`language`、`prompt_profile`、`evaluation_scope`、
`official_task`、`official_subtask`、`official_category` 和 `answer_choices`。
这些字段用于区分“使用官方评分算法”和“可与完整官方 split 直接比较”。

## VRSBench 转换约定

VRSBench 的每张原始图像会展开为 caption、object referring 和 QA 样本。检测任务的
`answer` 是 JSON 字符串：

```json
{"label":"building","bbox":[0.0,0.2,1.0,0.8]}
```

目标边界框采用 `normalized_0_1` 的 `[x_min,y_min,x_max,y_max]`。来源格式必须在配置中
显式声明为 `normalized_0_1`、`percent_0_100`、`scaled_0_1000` 或 `pixel_xyxy`，不会根据
数值范围猜测。坐标转换后裁剪到 `[0,1]`。原始值、
裁剪值和是否发生裁剪分别保存在 `metadata.bbox_raw`、`metadata.bbox_clipped` 和
`metadata.coordinate_clipped`。

转换后的 VRSBench 图片路径以数据集根目录为基准，例如
`Images/Images_train/000001.png`。因此训练或评估配置中的 `image_root` 应设置为
VRSBench 根目录，而不是其 `Images` 子目录。

计数答案统一为 `{"count":2}`。转换器支持数字、英文数字和 no/none；无法可靠解析时
记录 `metadata.counting_unresolved=true` 并降级为普通 VQA，不伪造计数。

合并前的单目标 `{"boxes":[[...]],"labels":["..."]}` 可向后兼容并转换为上述单目标
schema；包含多个目标时不会静默取第一个。旧 detection 答案若没有可恢复的 bbox，则记录
`metadata.detection_unresolved=true` 并降级为 VQA，不会把自然语言答案配到强制 JSON prompt。

## Prediction JSONL

普通评测、量化和可靠性实验共享以下基础字段：

```json
{"id":"sample-id","task_type":"detection","prediction":"...","reference":"...","metadata":{},"inference_latency_ms":12.3}
```

实验可追加 `variant`、`backend`、`compression`、`fault_case`、`validation` 和 `recovery`，
但不得改变基础字段语义。
