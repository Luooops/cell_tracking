# Motion v2：完整视频关联与可验证的改进

本版本完成视频推理，不要求模型一次读取 30 帧。局部 Transformer 提供关联分数，整段视频只存候选边，再以全局一对一匹配生成统一轨迹。第一版仍不处理分裂，也不生成漏检帧的 mask。

## 研究依据与实现边界

1. [Trackastra，ECCV 2024，§2.4–2.5](https://arxiv.org/html/2405.15700v2)：局部重叠窗口预测，合并到全局候选图；训练采用共同几何变换与时间采样增强。本项目采用同一大方向，但不是复制其官方模型。其最终候选边主要连接相邻帧；本模型允许 dt=1..max_gap。
2. [How To Make Your Cell Tracker Say "I dunno!"，2025](https://arxiv.org/abs/2503.09244)：对线性分配型细胞追踪的不确定性进行专门量化和校准。因此当前导出的 association probability 只能用于候选排序，不能声称为校准后的正确率。本版本没有复现该论文的完整校准方法。
3. [Attrackt，ICCV 2025 workshop，官方实现](https://github.com/funkelab/attrackt)：无监督追踪和交互微调是进一步减少标注成本的方向。本项目已有部分身份监督，本轮优先保证未知标签处理、完整视频评价和监督训练，不引入未验证的伪标签或额外预训练模型。

这些文献支持设计方向，不能证明本实现已超过 Trackastra。后继损失、保守检测删除、edge-vs-null 融合是针对本项目问题的工程/研究选择，需要消融验证，不宣称为已有论文的精确复现或已建立的新颖贡献。

## 本轮变化

### 1. 完整视频窗口融合

每个检测使用视频全局行号。只汇总目标检测的完整候选回看时段都被包含的窗口，避免窗口边界让 null 或边概率出现虚假优势。

每条边的融合分数为 `mean(logit(source,target) - null_logit(target))`。这样不同窗口的分类归一化常数不会混入分数。然后在整段视频的目标候选前驱与 null=0 上进行一次 softmax。它与 Trackastra 的均值概率融合不同，属于本模型的设计选择。

推理窗口至少含 max_gap+1 帧；过密时先去掉多余未来或更早上下文，保留完整候选回看范围。如果最小范围也超过 max_tokens，明确报错。默认 2048；这是对象总数限制，不是每帧数量。局部注意力仍是密集 O(N²)，没有声称已改为稀疏 Transformer。

### 2. 全局稀疏路径覆盖

候选边权重为上述 edge/null 对数比。阈值过滤后，使用 SciPy sparse minimum-weight full bipartite matching 和每个源对象自己的 dummy 目标，最大化总正边权。

每个检测最多一个前驱、一个后继；前向时间边保证无环。整段视频不分配 N×N 全局矩阵；内存主要取决于候选边数。匹配只优化定义的边权，不能把这种优化解释成真实轨迹一定正确。

### 3. 后继辅助监督

原父节点分类只让前驱在每个目标中竞争。新增可选 end 头与后继交叉熵，让同一源对象的候选后继在训练时也竞争。

只对已验证正连接的源对象提供后继目标。不从窗口边界、未知身份或缺少后继推断真实终止。损失为 `parent CE + 0.1 motion NLL + successor_weight successor CE`，默认 successor_weight=0.1，属于待调节超参数。

旧权重没有 end 头，模型构造默认 use_successor=False，因此仍可加载。统一训练配置默认启用该头，需要重新训练才能获得这一部分的改进。

### 4. 漏检与几何增强

默认随机删除约 10% 可监督检测；-2 检测保持，以免删除未知上下文产生假监督。后续对象沿原已验证父链寻找最近保留前驱。新间隔/距离超门限或父链可信度中断时设 -2，不设假 birth；如果整个增强样本失去监督则回退原样本。

同一窗口共享旋转/镜像，位置与运动方向一起变换；面积、轴长、偏心率、实心度和亮度特征在这些刚性变换下保持不变。验证集不增强。

该增强模拟检测缺失，不模拟合并/过分割，也不能证明能恢复任意长时间缺失。max_gap=3 只允许中间最多两帧漏检。

### 5. 资源和追溯

对象对 MLP 只计算合法候选边，避免分配 N×N×width 的无效对象对隐藏特征；有效边的计算等价于原密集实现。空间/时间注意力仍使用密集矩阵。

训练样本保存在 CPU，每步只把当前窗口搬到 GPU。训练只拟合训练特征统计，校验 NPZ 特征顺序和门限，权重记录特征定义与训练配置。history 增加前驱准确率、跨漏检前驱准确率和有效 gap 目标数；它们不等于全局轨迹指标。best.pt 仍按验证总损失选择。

## 服务器运行

```bash
cd /projects/u6yk/yitong/cell_tracking

python -m models.motion_tracking.prepare_npz \
  --data-root ./data/data \
  --output-root ./outputs/motion_tracking/npz_v2 \
  --window-frames 6 --stride 3 --max-gap 3 --distance 60 --max-tokens 2048

python train.py --model motion_tracking --version motion_v2 \
  --train-dir ./outputs/motion_tracking/npz_v2/train \
  --val-dir ./outputs/motion_tracking/npz_v2/val \
  --device auto

# 先用验证孔检查门限/全局指标；不要用测试孔选择参数。
python test.py --model motion_tracking --version motion_v2_val \
  --weights ./outputs/motion_tracking/motion_v2/train/checkpoints/best.pt \
  --wells r08c13 r11c12 r03c12 r14c13 --device auto

# 固定训练与门限后运行保留的三个测试孔，处理整段视频。
python test.py --model motion_tracking --version motion_v2 \
  --weights ./outputs/motion_tracking/motion_v2/train/checkpoints/best.pt \
  --device auto
```

目录必须使用新版本/新输出名。CTC mask 由现有 Cellpose 数据流程产生，本入口不分割图像。其已知 ID 仅用于训练标签和 CTC 评价；推理模型只有坐标与 7 个特征。

每孔输出：instance_tracks.csv（实际 mask instance_id 与全局 track_id）、frames.csv、summary.json、edges.json（全局检测行号）、association_candidates.csv（概率、融合次数和是否选中）、metrics.json、visualizations/*_tracking.png。CTC 指标复用项目 perfect/id_switches/purity，不等同于官方 TRA 或 XML 指标。

## 直接使用现有 CTC GT 评价

motion 推理不再使用 XML，也不需要原图名称或 UUID 映射。GT 来自 `<well>_GT/TRA/man_track*.tif` 的身份标签，以及 `man_track.txt` 的 id/start/end/parent。检查已知 ID 的掩膜观察是否在轨迹表区间内，且首末观察与表一致；当前仅支持无分裂数据。标签 >=50000 为本项目的未知身份扩展，不计入身份评分。

完整视频 test 自动输出 `<test>/<well>/evaluation/`，包括 metrics.json、per_gt_track.csv、transitions.csv、instance_matches.csv、per_frame.csv、visualizations/*_comparison.png。原 `<test>/<well>/metrics.json` 与根 metrics.json 仍保存汇总，原 tracking PNG 仍在 `<test>/<well>/visualizations/`。

可以对已经保存的完整视频预测独立 evaluate，不加载权重、不重新跑模型：

```bash
python evaluate.py \
  --data-root ./data/data \
  --tracking-output ./outputs/motion_tracking/motion_v2/test \
  --output-root ./outputs/motion_tracking/motion_v2/evaluate \
  --wells r16c02 r01c20 r07c07 \
  --visualize all
```

输出目录必须是新目录。省略 wells 时从 tracking-output 的直接子目录发现孔位。visualize=none 可只算指标。

评价要求预测来自同一批 CTC 检测：CSV 的 frame_index/instance_id 映射到 GT mask 中的对象，track_id 则是预测身份。G:1 对应 P:101 只要整轨迹一致且没有混入其他 GT，就可以正确；不能直接比较两种 ID 的数值是否相等。CSV 中若含 mask_path，则检查它指向对应 CTC mask。

指标包括 perfect（百分比）、id_switches、purity，以及有效 GT 观察覆盖率、混用预测轨迹数和按观察帧间隔 dt 分组的 identity_link_recall。后者衡量两个连续 GT 观察是否保持同一预测身份，不等同于直接边的准确率。未知检测可以作为追踪上下文和连接的中间节点，不会被当作已知错误身份。

比较 PNG 在同一原图上画 GT mask 边界与 G/P 编号：绿色表示整条观察轨迹恢复，橙色碎片化，紫色身份混用，红色缺少预测，灰色身份未知。颜色按整条轨迹质量判断，所以有问题的轨迹在早期帧也可能呈橙色或紫色。

这是现有检测上的关联评价。轨迹表跨越的内部无 mask 帧不会被虚构成 GT 检测；它不能衡量原始人工标注中被 Cellpose 漏掉的细胞，也不是官方 CTC TRA、SEG 或 CT 的实现。官方定义见 [CTC Evaluation Methodology](https://celltrackingchallenge.net/evaluation-methodology/)。

## 最小消融与判断标准

所有实验使用相同孔位划分、全局推理和验证选出的门限，至少三个随机种子。禁止把同孔重叠窗口拆到训练和验证两边。当前划分只能评估同批实验跨孔泛化。

| 实验 | use_successor | detection_dropout | geometric_augmentation |
|---|---|---|---|
| 基线 | false | 0 | false |
| 后继辅助 | true | 0 | false |
| 漏检增强 | false | 0.1 | false |
| 几何增强 | false | 0 | true |
| 完整模型 | true | 0.1 | true |

通过 train.py 的 `--set use_successor=false --set detection_dropout=0 --set geometric_augmentation=false --set seed=42` 设置消融。另用相同权重对比验证选择的不同 threshold；修改 max_gap 必须重新准备监督窗口和训练，不能只在测试时改权重配置。

报告完整轨迹恢复率、ID switches、purity，以及 dt=1/2/3 的已验证关联准确率、时间和峰值显存。增加人工漏检实验时删除输入检测但保留评价身份，分别统计跨缺失后的身份保留和检测覆盖率。关联不能补回缺失检测的覆盖率。

只有这些实验完成，才能说优化提高了准确率。当前回归测试证明实现契约和整段推理可运行；合成几何评分器的正确追踪不是学习模型的性能证据。

## 本轮实际验证（2026-10-09）

- `python -B -m unittest discover -s tests`：30 项通过，涵盖 30 帧身份延续、连续两帧漏检、完整候选覆盖、稀疏全局分配、删除检测后的标签重映射、后继损失梯度、合法边 MLP 与密集参考一致、原 XML 评价与 comparison PNG，以及其他模型回归。
- 真实数据运行验证：从既有 NPZ 取 r12c13 两个训练窗口与 r11c12 两个验证窗口，用 width=32/layers=1 在 CPU 训练 1 轮，再对 r11c12 全部 30 帧推理。5873 个检测通过 25 个窗口处理，输出全局 CSV、指标及 30 张 PNG。
- 上述微型训练的全局结果仍高度碎片化（5617 条预测轨迹，标注轨迹恢复率约 4.78%），因此只证明真实数据流程可运行，不能用来判断完整训练效果或宣称超过基线。它没有使用保留测试孔训练或调参。
- 临时训练权重、数据副本和 PNG 验证后清理，保留真实原始数据和已有 outputs。算法指纹只更新本轮明确改变的 MotionAssociation、association_loss、load_windows；其他模型及旧局部 decode_tracks 的指纹保留。

测试清理后只保留 motion 相关测试；上述 30 项为清理前的历史验证记录。历史 AST 指纹检查及其他模型测试已按用户要求移除。

当前 CTC 评价更新已移除 motion 的 XML/清单输入分支，增加独立 evaluate.py 入口。上述 XML 测试与 30 项测试数为旧版本历史记录。

本轮 CTC 修改验证：全套 23 项测试通过；真实 r01c20 的 30 帧、2394 个已知身份观察和 43 个未知身份检测，使用已保存的经典算法 CSV 验证独立评价入口。此检查不重新训练/评估 motion 性能，也不修改已有预测或 GT。
