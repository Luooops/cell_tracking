# Motion NPZ 数据准备

`python -m utils.prepare_motion_npz --output-root outputs/motion_tracking/npz_v001`

数据适配脚本位于 utils/prepare_motion_npz.py；不修改模型、损失、训练循环或解码算法。原图和 mask 只读。默认数据源为 data/data，输出目录必须不存在，拒绝覆盖或写入原数据目录。

## 数据划分

- train：r15c12、r12c13、r07c02、r09c01、r13c12、r16c13、r09c09、r14c03。
- val：r08c13、r11c12、r03c12、r14c13（原 GNN fold 0）。
- 原测试孔 r16c02、r01c20、r07c07 禁止进入生成的训练/验证集。
- 可使用 --train-wells 和 --val-wells 明确指定其他互不重叠的孔位。
- dataset_id 使用 `<dataset-name>/<well>`，--dataset-name 默认 lab_cells。整个孔位固定在同一集合，所有窗口共享同一 ID。

这是明确的孔位级划分，现有训练器据此检查分组重叠。由于原数据来自同一批实验，此验证只能说明同实验内跨孔的表现，不能据此声称跨实验泛化。不得给同一个源孔位换名字后分别放入训练和验证集。

## 特征

coords 为 float32 [N,3] 的 (原帧号,x,y)。features 为 float32 [N,7]，依次为 log1p(面积)、log1p(长轴)、log1p(短轴)、偏心率、实心度、mask 内平均亮度、mask 内亮度标准差。

亮度由每张完整原图的 1%/99% 分位数归一化到 [0,1] 后统计。分母最小 1e-6；标准差使用总体标准差。逐帧规则固定，不跨数据集拟合参数；训练集特征 z-score 仍由原训练脚本计算并保存在 checkpoint 中。以后处理推理图像必须复用相同特征定义。

GT 标签只用于制备 parents 和追溯，不作为输入特征。NPZ 额外保存 gt_labels、source_indices、feature_names、max_gap、distance，原训练器只读取其原有四个字段。

## 窗口与标签

默认 window-frames=6、stride=3、max-tokens=1024。窗口过密时按完整帧缩短，步长同步缩小以保留至少一个重叠帧。不会随机丢弃检测或空间裁剪。若连续两帧仍超过上限，明确报错；改变上限时必须匹配训练的 --set max_tokens 值，并评估显存。

parents[j] 是当前 NPZ 内的行索引，不是 GT ID。使用整个孔位的已知 GT 身份寻找最近检测前驱，然后映射到窗口局部索引：

- 标签 >=50000：未知身份，设为 -2。
- 前驱在窗口内、dt<=max_gap、距离<=distance*dt：可生成正标签。
- 两个已知检测之间若存在满足目标空间/时间门限的身份未知检测：最近前驱存在歧义，保守设为 -2。
- 已知前驱因时间窗口被截断，或真实连接超出距离门限：设为 -2 并记录原因，绝不改成“无前驱”。
- 只有窗口覆盖完整的候选回看时段，且其中不存在已知身份前驱或可能的未知前驱，才设为 -1（候选范围内无已观察到的前驱，并不代表生物学新生）。
- 窗口边界无法判定的对象设为 -2；没有任何有效监督的窗口跳过并记录。

默认 max-gap=3、distance=60，与 config.json 的 motion 训练值一致。修改生成门限时，训练端必须使用相同值。

该策略保留所有选中帧的未知检测作为上下文。监督可信度仍取决于已清洗 GT 的身份正确性；脚本不会修复原有身份切换或凭空确认缺失标注。原模型不处理分裂，含非零亲本的 man_track.txt 被拒绝。

## 输出与训练

```text
npz_v001/
  manifest.json
  train/<well>/t000_t005.npz
  val/<well>/t000_t005.npz
```

manifest.json 记录参数、特征定义、分组单位、每个窗口实际帧数/检测数及所有监督过滤原因。重叠窗口的统计是窗口内出现次数，不是去重后的检测数量。只有 status=complete 才表示整个转换成功；失败目录不会被自动删除或覆盖。

```powershell
python train.py --model motion_tracking --version v001 --train-dir outputs/motion_tracking/npz_v001/train --val-dir outputs/motion_tracking/npz_v001/val
```

默认网络 width=128、layers=3 保持不变，训练 30 轮。每个窗口独立训练和解码；本脚本不增加完整视频窗口融合。

验证命令：`python -B -m unittest tests.test_prepare_motion_npz tests.test_preservation`。包含跨漏检前驱索引、窗口截断、未知检测歧义、门限排除、超限窗口以及原模型损失和反向传播兼容性。
