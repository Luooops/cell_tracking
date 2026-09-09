# 基于位置的 Tracking 评估

本工具面向轮廓粗略、主要标注细胞位置和 track_id 的 GT。无需 GT mask，也不按 mask IoU 评分。使用已有 `torch` 环境；依赖 numpy、scipy、Pillow、tifffile，复用项目的 GT XML/ZIP 读取工具。

## CLI

从项目根目录运行：

```powershell
& D:\MiniConda\envs\torch\python.exe track_eval/evaluate.py --gt "E:\aws_gt_data\4cbffe68-5969-4f0a-8be6-9e02efd9720f\r16c13" --max-distance 45
```

在 track_eval 目录下可运行 `python .\evaluate.py --gt "GT路径"`。GT 支持普通 XML、ZIP（含伪装成 .xml 的 ZIP），或仅含一个 XML/ZIP 的目录。默认选取标签为 cell 的 polygon，track_id 从 polygon 的 attribute 读取。当前支持 per-image XML，不支持另一种以 `<track>` 为主体的插值格式。

## 预测定位

从 GT 路径提取数据 UUID（也可通过 `--dataset-id` 指定），在项目 `main_tracking/outputs/<UUID>__*/` 下搜索完整的 frames.csv、instance_tracks.csv、summary.json。只接受 summary 与帧状态 complete 的预测。根据 GT 完整图像文件名匹配，严格保持孔位、视野、层面和通道，不自动改通道。

找不到结果会输出 `no tracking predictions found.`、退出码 2，并按 GT 通道打印运行 main_tracking/main.py 的命令。建议命令中的输入目录默认取 GT 所在目录；若 GT 单独存放，请用 `--data-dir` 指定真实原图目录。已有非空预测目录不会由评估脚本删除或覆盖。

部分 GT 帧缺少预测时默认停止并列出缺失数量；`--allow-missing` 允许部分评估，缺失帧明确记录，不能解释成分割漏检。`--predictions-root` 可改预测根目录。

## 位置匹配

GT 位置使用多边形面积质心，退化多边形使用顶点均值；预测使用 CSV 中 mask 的质心。所有坐标都是原始像素，无配准或缩放。检查 GT 与预测图像尺寸。

按欧氏距离生成候选，在 `--max-distance` 范围内做一对一 Hungarian 匹配：先最大化有效配对数，再最小化距离和。默认 45 像素只是初始值，不是已校准阈值，建议人工抽查后比较 30/45/60 等设置。位置匹配不利用任何预测轨迹 ID，避免掩盖身份错误。

任一匹配端在门限内有多个候选，则标为 ambiguous；包括多个 GT 竞争同一预测。歧义匹配仍保留在完整结果中，但不参与无歧义相邻关联指标。这是保守筛选，不意味着其余匹配绝对正确。粗略标注中心偏差仍需要人工检查。

## 逐轨迹连续性

按序列和 GT track_id 分组，再按文件名时间排序：

- coverage：匹配观察数 / 全部 GT 观察数；coverage_available 仅以存在预测文件的 GT 观察为分母。
- distinct_pred_ids：同一 GT 轨迹对应了多少个不同预测 ID。
- matched_fragments：被未匹配 GT 观察分开的连续匹配片段数；仅 ID 改变不新增此片段数。
- id_changes_between_matched_observations：相邻成功匹配观察间的 ID 变化，可以跨过未匹配观察；是诊断计数，不称作标准 IDSW。
- adjacent_id_changes：连续两次 GT 观察均匹配时的 ID 变化，不跨过未匹配观察。
- unambiguous_pairs / unambiguous_id_changes：上一项中两个端点都没有位置歧义的配对数/ID 变化数。
- summary 的 unambiguous_association_accuracy：1 - 无歧义 ID 变化数 / 无歧义配对数。分母为 0 时是 null，必须结合覆盖率和配对数解读。

连续 GT 观察不一定是相邻原始帧，transitions.csv 中 time_gap 明确记录。GT 没有 track_id 的对象参与位置匹配，不参与轨迹指标。同帧多个 GT 对象使用相同 track_id 时，整个序列中该 GT ID 的轨迹被标为 gt_id_conflict 并排除连续性评分，位置匹配仍保留；summary 列出排除 ID，gt_id_conflicts.csv 保存相关对象。不会把未标注的预测细胞算作误检，不计算标准 IDF1/完整性未知的 precision。当前不评价分裂谱系。

## 输出和可视化

每次新建 `track_eval/outputs/<UUID>__<时间戳>/`，避免覆盖实验：

| 文件 | 内容 |
|---|---|
| position_matches.csv | 每个 GT 对象一行，双方 ID/坐标、距离、候选数量、歧义、预测是否存在 |
| per_frame.csv | 每帧匹配数、未匹配和歧义数，以及实际预测目录 |
| per_gt_track.csv | 每条 GT 轨迹的覆盖与连续性诊断 |
| transitions.csv | 连续 GT 观察的双方预测 ID、时间差、是否可评估和是否保持 ID |
| summary.json | 汇总、门限、来源、缺失帧和评估限制 |
| visualizations.csv | 图片路径及是否使用原图背景 |
| visualizations/ | 默认保存有歧义或未匹配 GT 的帧 |

可视化中 G 表示 GT ID，P 表示预测 ID；GT 轮廓和配对连线：橙色=歧义、红色=未匹配、绿色=已匹配；预测中心/ID 为青色。`--visualize all` 输出全部已配对帧，`none` 关闭。按预测 summary 的原图路径读取背景，`--data-dir` 可提供新位置；找不到原图则画在空白画布并记录，不影响数值评估。

存在 GT 身份冲突的帧也会进入 issues 可视化，冲突轨迹显示为洋红色。position_matches.csv 中 ambiguous 与 gt_id_conflict 是独立标记。

这是用于审查 tracking 的位置诊断工具，歧义率高时应先校准门限和抽查匹配，再解释 ID 变化。

回归验证：`python -m unittest track_eval.test_evaluate`（在项目根目录运行）。
