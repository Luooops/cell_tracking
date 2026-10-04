# 完美轨迹比例与 ID 切换（`perfect_tracks.py`）

`track_eval/` 里的第二个评估脚本，独立于 `evaluate.py`，不修改它的任何行为。只输出两个主指标：

| 指标 | 定义 |
|---|---|
| **完美轨迹比例**（perfect_frac） | 一条 GT 轨迹的所有已匹配检测都属于**同一个**预测 `track_id`，并且这个预测 `track_id` 里没有混入任何其他 GT 轨迹的检测，才算完美。分母是至少有一个检测被匹配上的 GT 轨迹数 |
| **ID 切换**（id_switches） | 沿每条 GT 轨迹，相邻两次已匹配检测之间预测 `track_id` 发生变化的次数，对所有 GT 轨迹求和。跨过漏检帧的断开算一次 |

另外附带输出：纯度（每条预测轨迹里占比最高的 GT 轨迹所占比例的平均值）、检测召回率、有漏检帧的 GT 轨迹数。

## 和 `evaluate.py` 的区别

| | `evaluate.py` | `perfect_tracks.py` |
|---|---|---|
| GT 和预测怎么对应 | 质心位置，`--max-distance` 以内 | mask 重叠（规则见下） |
| 分割漏检的那一帧 | 这条 GT 轨迹不算完整跟踪成功 | 那一帧不计分，但前后必须仍是同一个 `track_id`，否则算一次 ID 切换，轨迹也不算完美 |
| 重复 ID | 排除整条 GT 轨迹 | 只去掉出现重复的那一帧里的这些多边形 |
| 只出现一次的 GT 轨迹 | 不计入主指标 | 计入 |

所以这个脚本回答的是"在分割器给出的检测上，关联做得好不好"：漏检只有在 tracker 没能把前后两段接回来时才扣分。
两个脚本的数字不能直接比较。

## 预测实例如何获得 GT 身份

和我们整理训练数据时用的清洗规则相同：

1. 同一帧里同一个 `track_id` 出现在 2 个及以上多边形上：这一帧里带这个 id 的多边形全部去掉。没有 `track_id` 的多边形忽略。
2. 多边形分两类：`source="manual"` 或顶点数 ≤ 8 的算**粗框**（人工画的方框），其余算**精确轮廓**。
3. 每帧做一对一匹配，重叠大的优先：
   - 先让精确轮廓认领实例：重叠面积 / 实例面积 ≥ 0.5；
   - 再让粗框在剩下的实例里选：重叠面积 / min(实例面积, 多边形面积) ≥ 0.5。
4. 同一条 GT 轨迹在相邻两帧之间移动超过 `--jump-px`（默认 300 px，按多边形面积质心计算），在这里切成两条 GT 轨迹（标注跳变，不是细胞运动）。

没有匹配到任何多边形的预测实例身份未知：不计分，也不影响它所在预测轨迹的纯度。

## 运行

从项目根目录运行。输入是 GT 的 XML/ZIP（或只含一个 XML/ZIP 的目录）和预测结果目录；
预测目录需要 `frames.csv`、`instance_tracks.csv` 和 mask，`main_tracking` 与 `trackastra_tracking` 的输出都可以直接用。

```powershell
# 评估一个序列
python track_eval/perfect_tracks.py --gt "<XML 或 ZIP>" `
  --sequence-dir "main_tracking/outputs/<UUID>__r16c13/f01__p01__ch02"

# 评估某个输出根目录下、图像名出现在该标注里的所有序列
python track_eval/perfect_tracks.py --gt "<XML 或 ZIP>" --predictions-root trackastra_tracking/outputs
```

每个序列在终端打印一行汇总，并写出两个文件（默认写在该序列目录里，`--output-root` 可改）：

- `perfect_tracks.json`：汇总指标和计数；
- `perfect_tracks_per_gt_track.csv`：每条 GT 轨迹一行（匹配到的检测数、首末帧、是否有漏检帧、对应了几个预测 id、ID 切换次数、是否和别的 GT 共用预测 id、是否完美）。

依赖：numpy、tifffile、scikit-image。不需要 torch。

## 示例结果

本仓库自带的 r16c13 序列（`main_tracking` 的 cpsam_v2 mask，GT 为 2026-10 重新标注的版本）：

| | 完美轨迹比例 | ID 切换 | 纯度 |
|---|---|---|---|
| 经典算法（`main_tracking/outputs` 中的现有结果） | 19.28%（91/472） | 1248 | 0.924 |
| Trackastra（`trackastra_tracking`，general_2d + greedy_nodiv） | 38.77%（183/472） | 377 | 0.940 |

两者用的是同一批 mask：检测召回率 89.9%（6792/7555），472 条 GT 轨迹里有 150 条至少被漏检一帧。

## 验证

- 匹配规则：对 r16c13 用同一批 Cellpose mask，本脚本和原清洗脚本匹配到的检测数（7197）、去掉的重复 id 多边形数（18）、
  每一帧带身份的检测数都完全一致。
- 指标：同一份 tracking 结果，本脚本和我们原来的评估代码给出相同的数字（完美轨迹 38.37%，ID 切换 341，纯度 0.937）。
