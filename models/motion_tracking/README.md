> Historical model documentation. For current commands and paths, use the root README.md.

# Trackastra 流程与运动关联模型原型

本目录是独立的研究模型核心，不依赖 trackastra 包，不改变现有官方基线。
用户确认第一版只做身份关联和漏检恢复，不训练细胞分裂。
已实现前向计算、监督损失、跨帧无分裂解码、NPZ 训练入口与回归测试。
尚未完成真实 XML 训练标签转换、整段视频滑窗融合和项目 CSV 导出，
也没有真实数据训练结果。新结构不兼容 Trackastra 预训练权重。

## 1. Trackastra 原模型的输入

原论文采用 tracking-by-detection：外部分割模型先产生实例 mask，Trackastra
负责关联。当前项目使用 main_tracking 的 Cellpose mask，重新追踪时不重新分割。
原图用于计算对象亮度等特征。原始模型没有直接把整张显微图像交给 Transformer。

窗口默认覆盖 6 帧。若每帧有 100 个对象，则窗口约有 600 个 token；
token 是某个对象在某一帧的检测实例，不是整条轨迹，也不是像素 patch。
N 随细胞数量变化，训练时可以使用裁剪和 padding。

每个 token 保存时间 t、中心 p、面积、平均亮度和惯性张量等浅层特征。
惯性张量描述空间形状与方向。面积、亮度和形状是测量值，不是身份标识。
不能把 main_tracking 已预测的 track_id 作为模型输入，否则会混入基线的关联决策。

## 2. Token 编码与上下文

用可学习 Fourier 编码表示位置/时间，拼接对象特征后线性投影：

    X_i = W [PE(t_i, p_i), z_i]

当前官方代码还支持对低维特征进行 Fourier 编码。
原论文实验配置 d=256，encoder/decoder 各 6 层，4 个 attention head。
这些是论文配置，不代表每一个发布权重的配置完全相同，应以 config.yaml 为准。

Encoder 的自注意力得到对象的上下文表示 Y；decoder 以输入表示为 query，
以 encoder 表示为 key/value 做交叉注意力，得到另一组表示 Z。
这个 decoder 没有逐个生成 track_id，也没有维护完整视频的轨迹记忆。

    Attention(Q,K,V) = softmax(QK^T / sqrt(d_head) + mask) V

RoPE 把相对时空位置注入 query/key；空间距离门限排除过远对象。
窗口包含后续帧，因此常规滑窗处理使用未来上下文，不应直接宣称是严格在线算法。
注意力矩阵和最终关联矩阵都具有二次规模；mask 不等于稀疏存储。

## 3. 关联头与训练目标

两套表示经过各自的 MLP，然后计算点积：

    S_ij = <MLP_Y(Y_i), MLP_Z(Z_j)>

S 是关联 logits，attention 权重和 S 不是同一张矩阵。
模型可以在时空上下文中隐式学习运动，但没有显式速度或运动不确定性预测头。

原论文监督目标 A_ij=1 表示两对象属于同一条祖先—后代链。
跨非相邻帧的同一细胞，以及母细胞与后代都可以为正。
同一母细胞的两个女儿是姐妹，彼此不是祖先/后代，不能简单标成同一身份。
检测实例与 GT 对象要先匹配，才能生成关联监督。

Parental softmax 对候选亲本进行竞争，quiet softmax 保留无亲本选项：

    P(i -> j) = exp(S_ij) / (1 + sum_k exp(S_kj))

简式描述上一帧候选亲本的归一化；官方实现还按时间块与前向/后向关系处理。
这个设计允许一个母细胞连接多个子细胞；最终离散约束仍由解码器保证。
归一化概率不是经过独立校准验证的置信度。

原论文损失为加权 BCE(parents-normalized probabilities) 加 0.01 倍
BCE(sigmoid logits)，并强调继续轨迹和分裂事件；默认监督时间间隔上限为 2。
类别加权不等于额外的独立分裂预测分支。

## 4. 从概率到轨迹

滑动窗口重复计算同一对对象的关联，然后融合预测。
原论文最终主要选择相邻帧的候选边，使用贪心或 ILP/LAP 解码。
允许分裂时每个对象最多一个亲本和两个女儿；greedy_nodiv 只允许一对一。
大概率候选边也可能因冲突被解码器拒绝。

因此，增大窗口不自动等于支持漏检恢复。候选边、训练目标、解码器都必须
显式支持非相邻帧连接。连接漏检前后检测也不意味着补出了缺失的 mask。

参考：
- 原论文：https://arxiv.org/abs/2405.15700
- 官方模型：https://github.com/weigertlab/trackastra/blob/main/trackastra/model/model.py
- 编码/注意力：https://github.com/weigertlab/trackastra/blob/main/trackastra/model/model_parts.py
- 归一化：https://github.com/weigertlab/trackastra/blob/main/trackastra/utils/utils.py

## 5. 新结构：MotionAssociation

    已有检测 + 特征
        -> 位置/时间编码与特征投影
        -> 同帧空间 attention
        -> 跨帧时间 attention（交替重复）
        -> 上下文点积 + 显式对象对特征 + 运动残差
        -> 跨时间间隔前驱竞争 + 可学习 null
        -> 最大权重无分裂路径覆盖

### 结构与假设

默认宽度 128、3 个 block、4 个 head。较小配置用于验证，不是已证明的最优配置。
同帧 attention 聚合邻域信息；跨帧 attention 聚合时间上下文。
每层保留自连接，避免孤立对象出现全 mask softmax。
空间坐标按窗口中心平移；时间按窗口最早时刻平移。

评分由三项组成：

    score = contextual_dot_product + pair_MLP - motion_error

pair_MLP 输入包含 dx、dy、dt、距离和标准化对象特征差。
标准化后的特征差不是原始面积比，也不应混淆。
运动头输出每帧二维速度 v_i 与标准差 sigma_i；目标位置预测为 p_i + v_i * dt。
概率评分使用高斯运动残差与 log(sigma)，避免无限增大 sigma 来逃避运动惩罚。
sigma 是待学习的误差尺度，不是已经证明校准的生物学不确定性。
使用常速度/误差随 sqrt(dt) 增长的简化假设，快速形变或转向仍可能失败。

候选边允许 1 <= dt <= max_gap，空间门限为 distance * dt。
默认 max_gap=3 表示最多跨越两个没有检测的中间帧。
门限是模型候选门限，与 evaluate 的 max-distance 不是一个概念。
门限之外的真实连接不能被模型恢复，训练时会报错而不是静默改标签。

所有过去时间间隔的候选前驱与一个可学习 null logit 共同 softmax。
不同于原模型的祖先关联目标，这里每个对象只选择一个最近的已有检测前驱。
null 的含义是“当前候选窗口内无已观察到的前驱”，不一定是生物学新生。

解码将 log(P_edge/P_null) 作为收益，在超过概率门限的前向边上
求一对一最大权重匹配，相当于允许对象不匹配的无分裂路径覆盖。
一个对象可以同时有一个前驱和一个后继；时间严格递增，所以不会形成环。
允许 dt>1 的边，但不插入虚拟检测，也不删除已有检测。
这比纯贪心增加了优化成本，不能未经测量声称更快。

### 标签与损失

parents[j] 必须是“已确认的最近检测前驱”的窗口内索引：
- >=0：可靠前驱索引。
- -1：已确认候选窗口内无前驱。
- -2：未知，不对该对象的前驱选择计算监督损失。

不能把 XML 缺少 track_id 的对象一律标成 -1；不能把间隔多个检测的祖先
直接标成最近前驱。窗口截断、漏标和真正漏检要区别处理。
某对象拥有已验证唯一前驱时，其他候选可作为竞争项；未知目标整项忽略。
未知对象仍参与上下文计算，忽略监督不等于从图中删除。

损失 = 前驱分类交叉熵 + 0.1 * 已确认连接的运动高斯 NLL。
运动消融会同时关闭运动评分与运动辅助损失。
训练使用同一份真实预测 mask 最有利于对齐部署分布，但仍需独立验证。

### 仍然不是已验证的新方法

分解注意力、相对几何、运动先验、可学习 null、路径覆盖各自已有广泛研究。
这里实现的是组合研究假设，不能仅凭这些模块宣称论文创新。
必须与 Trackastra、微调 Trackastra、SAM2 特征版本及简单 gap closing 对照。
新结构无法直接加载 Trackastra 权重，需要自身训练或另行设计蒸馏。
当前仍使用 dense attention 和 N*N 输出，内存复杂度 O(N^2)；真正的稀疏实现另需开发。

## 6. 训练数据契约与 CLI

每个 NPZ 保存一个经过审核的窗口（建议先 6 帧）：

| key | shape | 含义 |
| --- | --- | --- |
| coords | N,3 | float32，time,x,y；time 是整数帧号；坐标单位像素 |
| features | N,F | float32；所有文件采用相同列定义，输入未标准化特征 |
| parents | N | int64，按以上三类标注 |
| dataset_id | scalar string | 实验 UUID，用于阻止训练/验证交叉 |

建议首版固定 7 个对象特征：log(1+area)、log(1+major_axis)、log(1+minor_axis)、
eccentricity、solidity、归一化原图上 mask 内平均亮度、亮度标准差。
亮度归一化方式要在训练和推理一致；不使用 tracker 已预测的 ID 或轨迹长度。
全局特征均值和标准差只从训练集拟合，保存到 best.pt，验证/推理复用。
不要将同一个实验换不同 dataset_id 来绕过检查。
XML 尚未转换，以下命令只能在准备好符合上述契约的窗口后执行：

```bash
cd /projects/u6yk/yitong/cell_tracking
python models/motion_tracking/train.py \
  --train-dir models/motion_tracking/data/train \
  --val-dir models/motion_tracking/data/val \
  --output models/motion_tracking/runs/experiment_01 \
  --max-gap 3 --distance 60 --epochs 30
```

只依赖项目已有 PyTorch、NumPy、SciPy；不要求安装 Trackastra。
每个窗口单独优化，训练入口使用普通精度，未优化吞吐/显存。
默认 max-tokens=1024；超限要做带标签重映射的空间裁剪，不可静默截断。
输出 best.pt 和 history.json；选择依据是验证损失，不代表 evaluate 指标最优。
需在完整视频上另行选择检查点/解码门限并测量轨迹恢复率。

消融设置：
- --no-motion：无显式运动项，仍有上下文和几何。
- --no-pair-features：无对象对 MLP，仍有上下文和运动。
- --max-gap 1：只连接相邻帧，区分跨漏检连接的贡献。

## 7. 有意义的验证与实验

```bash
python -B -m unittest motion_tracking.test_model
```

测试包含：概率归一化、无效边屏蔽、空/孤立对象、平移与置换一致性、
未知标签忽略、非法监督门限检查、跨漏检帧解码、微型样本拟合、
训练命令与权重读回、实验 ID 泄漏拒绝。微型拟合不能当作真实性能。

下一步应先审核原始/修订 XML 与预测实例的身份对应，按实验 UUID 分组，
生成训练窗口，随后接入完整视频滑窗融合和现有 CSV/evaluate 输出。
标注不完整时不可直接宣称未匹配对象为假阳性；继续使用项目指标并报告标注范围。
refined GT 删除了对象时同时报告删除数，并保留原始人工标签审计。
训练数据中随机删除检测来模拟漏检时，要同步将后续对象的前驱改为最近保留检测。
在未见实验上报告完整轨迹恢复率、身份混用、不同漏检间隔的恢复率、耗时/显存。
人工漏检实验仅删除输入检测，保留评价标签；缺失帧本身的检测覆盖率不会被连接补回。
