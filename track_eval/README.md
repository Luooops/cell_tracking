# 基于位置的 Tracking 评估

## 新版主指标：GT 轨迹完整跟踪成功率与辅助指标（schema_version=2）

### 指标显示名称（统一术语）

后续图表与汇报使用“成功率”，不再将 recovery 翻译为“恢复率”。这些指标以 GT 为分母，并非所有预测结果的精确率。

| 兼容字段名 | 正式显示名称 | 英文显示名称 |
|---|---|---|
| complete_gt_track_recovery | GT 轨迹完整跟踪成功率 | Complete GT track success rate |
| gt_link_recovery_conservative | GT 相邻连接成功率（含歧义） | GT adjacent-link success (including ambiguity) |
| gt_link_recovery_decidable | GT 相邻连接成功率（排除歧义） | GT adjacent-link success (excluding ambiguity) |

“含歧义”指歧义连接保留在分母、不计成功；“排除歧义”仍将未匹配连接保留在分母。CSV/JSON 字段、文件名和状态值保留原名以兼容既有分析脚本。判断规则和数值不变。历史导出图与 PPT 不会自动更新。

运行命令不变，每次评估自动输出 `metrics.csv`、扩展的 `per_gt_track.csv` 和 `statistics/`。统计图使用浅蓝、薄荷绿、淡紫等浅色，深色文字保证可读性；同时保存 PNG 和可编辑 SVG。图中文字用英文避免环境中文字体缺失。`--visualize none` 只关闭逐帧对比图，统计图仍生成。

### 评估范围与符号

轨迹身份以 `sequence + track_id` 为键。排除没有 GT ID 的对象和同帧重复 ID 涉及的整条 GT 轨迹后，称为有效身份集合。主轨迹指标进一步排除仅出现一次的 GT 轨迹，至少需要 2 次标注观察。

- N：所有 GT 对象的累计观察数，包含无 ID、冲突 ID 对象；只用于位置指标。
- M：成功位置匹配的观察数，包含歧义配对。
- R：成功且无位置歧义的观察数，位置指标不要求 GT 身份有效。
- T：至少出现 2 次的有效 GT 轨迹数。
- F：完整跟踪成功的 GT 轨迹数。
- L：有效 GT 身份在时间点相差恰好 1 的两次观察之间建立的连接，且两个预测帧文件均可用。
- U：L 中至少一个端点有位置歧义的连接数。
- C：L 中两个端点均成功、无歧义匹配的连接数。
- S：C 中两端预测 ID 相同的连接数。

### 完整跟踪成功的严格定义

每次 GT 标注观察都必须匹配到预测，所有观察无位置歧义、预测帧文件可用，且整条轨迹只对应一个预测 ID。该预测 ID 还不能在此序列其他时刻可靠地对应另一个有效 GT ID。混用检查使用所有有效 GT 身份，包括单帧 GT；不使用冲突身份或歧义配对判断混用。

这只证明在 GT 标注时刻恢复完整，不证明未标注期间正确。GT 范围以外的预测延伸不会自动判错；未标注细胞之间的身份混用不可观测。本实现没有计算标准 IDF1、MOTA 或 HOTA。

### 指标公式与含义

`metrics.csv` 每行均给出 metric、numerator（分子）、denominator（分母）、value（0～1）。JSON 分母为 0 时值为 null，CSV 为空，图上显示 N/A，不伪造 0%。

| 指标键 | 计算 | 含义 |
|---|---|---|
| position_coverage | M / N | GT 对象有多少被找到；红色计入分母，橙色成功配对计入分子。 |
| reliable_position_coverage | R / N | 仅计无歧义配对的位置覆盖率；不是身份正确率。 |
| complete_gt_track_recovery | F / T | **主要指标**：完整跟踪成功的有效 GT 轨迹比例。红色导致跟踪失败；有橙色观察的轨迹保留在分母、不计成功。 |
| conditional_identity_retention | S / C | 两端均可靠检测到时的身份保持率；只看 time_gap=1，不包含漏检连接。 |
| gt_link_recovery_conservative | S / L | 相邻 GT 连接的成功率（含歧义）；漏检和歧义都保留在分母，仅已确认正确进入分子。 |
| gt_link_recovery_decidable | S / (L-U) | 排除位置歧义连接后的连接成功率，仍保留红色漏检连接。 |
| ambiguous_link_fraction | U / L | 位置配对不确定的连接比例，解释保守值与可判定值差异。 |
| track_review_fraction | 待确认轨迹数 / T | 预测文件齐全但至少有一个歧义观察的轨迹比例；其中可能也存在漏检或换 ID。 |
| observed_prediction_mixing_fraction | 对应多个有效 GT ID 的预测 ID 数 / 有可靠 GT 身份配对的预测 ID 数 | 预测轨迹身份混用的诊断比例。分母仅含可观察的身份配对，不包含全部预测，不能称整体误检率。 |

S 是局部连接判断，没有要求整条预测轨迹都无身份混用；F 则有这一额外要求。因此局部连接保持很好，但整条轨迹仍可能跟踪失败。

### 紫、橙、红与缺失文件

- 紫色：整条 GT 轨迹排除身份指标，位置指标保留；summary 原字段记录排除 ID。
- 橙色：不删除观察。有歧义的轨迹主指标记为未确认成功，并保留在 T；相关连接进入 L/U，不进入 C/S。
- 红色：位置与轨迹完整跟踪分母保留；不能先删红色再声称完整跟踪成功。
- 预测文件缺失：默认停止。若显式 `--allow-missing`，整条轨迹仍留在 T，不计完整跟踪成功，状态为 missing_prediction；涉及缺失文件的连接不进入 L，并单独统计 excluded_missing_prediction_links。此时是部分评估，不能与完整预测结果直接排名。

### 每条轨迹状态和新增字段

`per_gt_track.csv` 新增 recovery_eligible、recovery_status、fully_recovered、has_ambiguous_observations、has_missing_predictions、prediction_identity_mixed、reliable_coverage。

状态按顺序判定：single_observation（单次观察、不入 T）→ missing_prediction（缺文件）→ needs_review（有歧义）→ recovered（满足全部完整条件）→ not_recovered（其他失败）。状态互斥，但原因可叠加，应结合独立标志、coverage 和 distinct_pred_ids 查看。紫色轨迹不进入该表，详见 gt_id_conflicts.csv。

`reliable_coverage` = 本轨迹无歧义成功观察数 / 本轨迹 GT 观察总数。既有 coverage 包含歧义成功配对。完整跟踪成功并非覆盖率达到 90% 即成功：本版严格要求 100%，未实现宽松版本。

### 新输出文件与图

| 文件 | 内容 |
|---|---|
| metrics.csv | 所有新版指标的分子、分母、数值 |
| recovery_by_length.csv | 按 2–9、10–19、≥20 次标注观察分组的 F/T，避免短轨迹掩盖长轨迹问题 |
| adjacent_links.csv | 仅 time_gap=1 的全部有效 GT 连接，包含缺文件、歧义、可靠匹配及连接成功标记；缺文件连接保留在表中但不进入 L |
| prediction_identity_mapping.csv | 每个有可靠身份配对的预测 ID 对应的 GT ID 集合及数量；gt_count>1 表示已观察到混用 |
| statistics/metric_overview.png/.svg | 位置覆盖、完整轨迹完整跟踪、条件身份保持及两种连接成功率；标注各自分子/分母 |
| statistics/track_recovery.png/.svg | 长度分组成功率及轨迹状态数量 |
| statistics/track_distributions.png/.svg | 有效多帧 GT 轨迹的位置覆盖率分布和不同预测 ID 数量分布 |

完整跟踪成功率按轨迹计权，位置覆盖率按对象观察计权，连接指标按连接计权；不能相乘推导一个总体正确率。主图各条的分母不同，请看数值旁的计数。

summary 的 `recovery_evaluation` 保存以上新指标及其计数。旧顶层 `unambiguous_association_accuracy` 和原 transitions.csv 保留兼容，其中“相邻”指连续 GT 观察，允许 time_gap>1；与新版仅 time_gap=1 的 conditional_identity_retention 不同，不能混用。跨间隔连接仍可在旧 transitions.csv 单独检查。

验证：`python -m unittest track_eval.test_evaluate track_eval.test_metrics`。覆盖完整跟踪成功、漏检、身份混用、歧义保守分母、单帧排除、稀疏观察和空结果。

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
| gt_id_conflicts.csv | 存在同帧重复 ID 的 GT 轨迹在所有帧中的对象；并非每行都发生了同帧重复 |
| summary.json | 汇总、门限、来源、缺失帧和评估限制 |
| visualizations.csv | 图片路径及是否使用原图背景 |
| visualizations/ | 默认保存有位置歧义、未匹配 GT 或 GT 身份冲突标记的帧 |

### PNG 中的文字、轮廓和连线

- `G:25` 表示 XML 多边形的 `track_id=25`，不是多边形在文件中的顺序号，也不是标注工具的对象编号。
- `G:?` 表示这个 GT 对象没有 track_id，仍参与位置匹配，但不参与身份连续性评分。
- `P:31` 表示预测 tracker 的 track_id=31。GT 和预测的编号独立，`G:25` 配到 `P:31` 完全正常，不要求两者数字相同。
- 彩色多边形轮廓来自 GT；青色小圆圈标出预测中心。当前评估 PNG 不绘制预测 mask 轮廓。
- 从 GT 中心到预测中心的线表示当前帧的位置配对，不是跨帧运动轨迹。两个中心很近时，连线可能几乎看不见。

### 颜色含义

| 颜色 | 判定条件 | 对评估的影响与阅读方法 |
|---|---|---|
| **绿色** | GT 已获得预测配对，且没有位置歧义或 GT 身份冲突 | 表示当前帧的位置匹配成功；**不代表跨帧 tracking 正确**。同一 GT 在两帧都显示绿色，但对应的 P 编号改变，仍可能发生身份切换。 |
| **红色** | GT 未获得预测配对，且没有更高优先级的歧义或身份冲突标记 | 可能是预测漏检、小目标过滤、GT 位置偏差或距离门限限制。红色本身不能确定原因，也不表示预测 ID 切换。 |
| **橙色** | GT 或其候选预测端在距离门限内存在多个候选，且没有 GT 身份冲突 | 是位置配对歧义，包括多个 GT 竞争同一个预测。对象可以已匹配，也可以未匹配。已匹配时保留配对结果，但涉及该对象的相邻关联不进入无歧义指标。 |
| **紫色／洋红色** | 该 GT ID 在同一序列的至少一帧被两个或更多 cell 多边形同时使用 | 整条 GT 轨迹在所有帧都标紫，并排除身份连续性评分；位置匹配和覆盖率统计仍保留。**当前帧不一定存在重复 ID**，也不能据此认定预测有错。 |
| **青色** | 预测细胞中心和 P 编号 | 只是显示预测对象；所有预测检测都绘制，包括未匹配到 GT 的检测。青色不表示正确或错误，未配到 GT 的预测也不直接算作误检。 |

GT 轮廓、G 文字及存在时的配对连线采用同一颜色。显示优先级为：**紫色 > 橙色 > 红色／绿色**。例如，一个对象既属于冲突 GT 轨迹，又未匹配到预测，显示为紫色而非红色。仅看颜色不能判断所有状态，应结合 `position_matches.csv` 的 `gt_id_conflict`、`ambiguous` 和 `pred_track_id`（空值表示未匹配）查看。

### 为什么某帧只有一个 GT 25，却仍显示紫色？

假设第 1～6 帧每帧各有一个 GT 25，第 7 帧出现两个不同多边形都写着 GT 25，第 8～30 帧又各只有一个 GT 25。当前脚本发现第 7 帧重复后，会将 **GT 25 在全部 30 帧中的对象**标为紫色，并排除整条轨迹的连续性评分。

这是脚本采取的保守排除规则，不意味着其他 29 帧也有重复，更不意味着整条轨迹的所有标注都错误。`gt_id_conflicts.csv` 同样记录整条受影响轨迹；要查真正重复的位置，需要在同一 `sequence + image_name + gt_track_id` 下找到多行，而不是把该 CSV 的每一行都当作重复。

因此应区分“实际同帧重复对象数”和“因整条轨迹排除而受影响的对象数”。紫色数量反映后者。当前代码不会自动修正、合并或删除 GT 标注。

### 如何查看跨帧 tracking 是否正确

例如，同一 GT 轨迹在 t01、t02 分别显示 `G:25 → P:31` 和 `G:25 → P:78`，两个对象可以都是绿色。这只说明两帧的位置配对都无歧义，P 编号的变化仍会在符合条件时记录为 ID 变化。具体查看 `transitions.csv`，整条轨迹汇总查看 `per_gt_track.csv`。当前 PNG 没有专门表示 ID 切换的颜色。

文件名的 `t07` 与 XML `image id="6"` 可能对应同一图像；核查时以完整原图文件名为准，不要混用文件时间点、XML 帧编号和预测内部帧序号。

### 可视化输出选项

`--visualize issues`（默认）输出存在位置歧义、未匹配 GT 或身份冲突标记的帧；`all` 输出所有有对应预测记录的 GT 帧，`none` 关闭。身份切换本身不会触发 issues 输出，如需查看完整时间过程请使用 `all`。

按预测 summary 的原图路径读取背景，`--data-dir` 可提供新位置；找不到原图则画在空白画布并在 `visualizations.csv` 记录，不影响数值评估。

这是用于审查 tracking 的位置诊断工具，歧义率高时应先校准门限和抽查匹配，再解释 ID 变化。

回归验证：`python -m unittest track_eval.test_evaluate`（在项目根目录运行）。
