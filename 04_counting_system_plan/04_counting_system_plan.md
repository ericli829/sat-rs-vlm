# Counting System 计划书

## 1. 任务详细定义

目标：

> 在高分辨率整图或指定 Region 中，对目标实例进行尽可能可靠的计数。

逻辑接口：

```python
COUNT(target: TargetSpec, entire: bool) -> ScalarInt
```

视觉输入来自 DAG `inputs`，不塞进 `params`。

## 2. 两种输入模式

### ImageRef / Region

```text
scope
→ ScalePolicy
→ tiling
→ LAE-DINO
→ local→global bbox
→ same-scale dedup
→ cross-scale fusion
→ count
```

### EntitySet

如果上游已经：

```text
LOCATE → SELECT → EntitySet
```

则：

```text
COUNT(EntitySet) = len(EntitySet)
```

不能重新检测。

## 3. entire=true

整张 UHR 图：

```text
whole image
→ optional coarse Retriever gate
→ native tiles
→ optional fine scale
→ LAE
→ fusion
→ count
```

CLIP/Retriever 是可选加速；没有它仍必须能完整运行。

## 4. entire=false

当前输入已被上游限定：

```text
Region
→ exhaustive detector pass
→ dedup/fusion
→ count
```

COUNT 在当前 scope 内必须是 exhaustive，不做 Top-K 搜索。

## 5. Scale Policy

第一版至少支持：

- Global：整 scope resize，用于大目标和全局参考；
- Native：主力 tile；
- Fine：针对 tiny object 的更高有效放大倍率。

Fine 是否开启由类别/尺寸 profile 决定，不用“当前置信度低”作为唯一依据。

## 6. LAE 接口

输入：

```python
DetectionRequest:
    image/region
    target
```

输出 detection：

```text
bbox_xyxy_global
score
label/target
tile_id
scale_id
provenance
```

## 7. 去重与融合

### 同尺度

推荐 Core Ownership：

- tile 有 ownership core；
- detection center 落在 core 才归该 tile；
- 再做全局 NMS 保险。

### 跨尺度

保守规则：

- coarse↔fine 一一对应 → 保留更细检测；
- 一个 coarse 内有多个 fine → coarse 可能是 aggregate，丢 coarse；
- coarse 无 fine match → 保留 coarse。

## 8. 阈值实验

必须保存 raw proposals。

建议 sweep：

```text
0.40 0.30 0.20 0.15 0.10 0.05 0.02
```

区分：

- 模型没生成 proposal；
- proposal 有，但被后处理阈值截掉。

## 9. Prompt profile

开放词汇 detector 需要测试：

```text
ship
ship . vessel
ship . boat . vessel
boat
vessel
```

给主要类别建立 prompt profile。

## 10. Optional coarse gate

用于 `entire=true`：

```text
coarse tile
→ low-threshold retrieval gate
→ survivor tiles
→ LAE
```

原则：recall first，不是 Top-K。

## 11. 输出规范

内部保留：

```python
CountResult:
    count: int
    detections: DetectionSet
    provenance: dict
```

TaskGraph 对外主要暴露 `ScalarInt`；完整 detection 放 trace。

## 12. 验收指标

- exact count accuracy；
- MAE；
- detection recall / precision；
- duplicate rate；
- detector calls；
- latency；
- peak VRAM。

## 13. 必做实验

1. 1333/1024/896 source scale；
2. threshold sweep；
3. prompt wording；
4. same-scale dedup；
5. cross-scale fusion；
6. whole-image vs tiled；
7. optional Retriever gate；
8. tiny-object 类别单独统计。

## 14. 最低验收目标

1. Fake E2E 通过；
2. 真实 LAE 可跑；
3. Region count 可跑；
4. UHR tiled count 可跑；
5. global bbox 映射正确；
6. 去重稳定；
7. overlay + JSON trace；
8. 在若干 MME/XLRS 真实计数样本上完成 benchmark。
