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
- `task_type`：支持 `detection`、`counting`、`captioning`、`vqa`、`scene_classification`、`change_detection`。
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
{"task_type":"detection","instruction":"请检测图像中的飞机。","answer":"检测到疑似飞机目标，位于机场跑道附近。"}
```

counting：

```json
{"task_type":"counting","instruction":"请统计图像中的建筑物数量。","answer":"图像中大约有 5 个建筑物。"}
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
