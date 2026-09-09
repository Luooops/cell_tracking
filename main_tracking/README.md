# 分割、Tracking 与 Overlay

## 运行

从项目根目录运行，使用已安装 Cellpose 且支持 `cpsam_v2` 的 Python 环境：

```powershell
& D:\MiniConda\envs\torch\python.exe main_tracking/main.py --input-dir "E:\aws_gt_data\4cbffe68-5969-4f0a-8be6-9e02efd9720f\r16c13"
```

也支持 `python -m main_tracking.main`；不传 `--input-dir` 时提示输入目录。`--help` 查看全部参数。依赖：cellpose、torch、numpy、scipy、scikit-image、tifffile、Pillow、pandas、matplotlib。首次使用模型可能下载权重；离线运行可以指定 `--model "本地权重路径"`。

默认输出到本脚本旁的 `outputs`，可用 `--output-root` 修改。输出根目录必须在输入目录之外。已有非空序列结果会报错，请为新实验选择新的输出根目录，避免覆盖或混入旧 mask。

本机验证使用 `torch` 环境。当前名为 `cellpose` 的环境缺少 pandas、scikit-image 和 matplotlib，不能直接运行整个流程；没有对该环境安装或修改依赖。

回归检查：`D:\MiniConda\envs\torch\python.exe -m unittest main_tracking.test_pipeline`。使用模拟分割模型验证真实 tracking、导出和 overlay，覆盖空帧、全空序列、短轨迹、分组和无效输入；不依赖下载权重。

## 1. 发现输入并建立帧清单

递归读取 TIFF/PNG/JPEG，要求文件名形如 `r16c13f01p01-ch01t01.tiff`。输入应为原始数据目录，不能混入 mask 或预览。当前仅支持单帧二维灰度；RGB、多页 TIFF、Z/T stack 会报错，需先明确选取通道或拆帧。

从路径提取数据 UUID；没有 UUID 则使用输入目录名。按数据、孔位、视野 f、层面 p、通道 ch 分组，每组独立跟踪、ID 从 1 开始。默认只处理 `ch2`（也匹配文件名中的 `ch02`），可用 `--channel ch01` 显式切换通道。相同分组内多个来源出现重复时间点会报错，不会自动拼接。

按文件名 t 的数值排序，允许从任意 t 开始，但要求时间点连续、无重复、图像尺寸一致。所有输入先校验再加载模型。`frame_index` 是从 0 开始的内部序号，`time_index` 是原始 t 编号，两者不能与 GT XML 的 image id 混用。

## 2. Cellpose 分割

显式加载 `cpsam_v2`，并检查当前 Cellpose 是否识别该模型名；本地权重路径必须存在。模型在整个批次只初始化一次。默认检测 CUDA，不可用时使用 CPU并打印设备；`--cpu` 强制 CPU。summary 记录实际权重路径和软件版本。

默认保留历史预处理：1%–99% 分位裁剪与归一化、sigma=80 的高斯背景扣除、gamma=1.8；之后 Cellpose eval 仍使用其默认 normalize。`--preprocessing cellpose` 跳过外部预处理，直接让 Cellpose 处理原图，便于比较预处理效果。

初始参数为 diameter=25、cellprob_threshold=1.8、flow_threshold=1.0，沿用既有实验值，并非已通过整个数据集验证的最优参数。

## 3. 小目标过滤与整数 mask

默认移除面积小于当前帧平均实例面积 × 0.25 的实例。用 `--auto-min-area-fraction` 调整比例，或 `--min-area 500` 指定固定像素面积，`--min-area 0` 关闭此后处理（不改变 Cellpose 内部默认最小尺寸设置）。

过滤仅在分割之后执行一次，随后实例标签固定。TIFF 背景为 0，正整数是帧内 instance_id，默认 uint16，超过 65535 则 uint32。mask 保持原图尺寸。绝不通过 8-bit 灰度归一化保存分析标签。

## 4. Tracking 与断轨连接

复用 tracking_v0 中的 tracker：提取质心、面积和形状；结合运动预测、距离、面积/形状约束和 bbox IoU，用分阶段 Hungarian 匹配关联细胞。max_lost 控制可容忍的连续漏检帧数。注意“没有检测到细胞的帧”与“缺失图像文件”不同：前者可处理，后者输入校验报错。

逐帧读取和分割，不将全部 mask 堆叠进内存；内存仍保存检测历史。最后通过 gap closing 合并满足距离、形状和方向条件的断轨。参数 `--gap-close-max-gap` 沿用底层函数的帧索引差定义（不是中间缺失帧数），`--gap-close-max-distance`、`--max-close-cost` 控制连接门限。

主入口保留所有轨迹，`--min-track-length 5` 仅生成 passes_min_track_length 标志，便于评估时保留短轨迹。底层 tracking_v0 的独立旧入口仍保留原有长度过滤行为。该 tracker 是一对一关联，没有实现细胞分裂母女谱系，也不会为漏检帧生成虚构 mask。

## 5. 映射与输出

```text
outputs/<数据UUID>__<孔位>/f01__p01__ch02/
  masks/<原图stem>_mask.tiff
  overlays/<原图stem>_overlay.png
  frames.csv
  instance_tracks.csv
  tracks.csv
  summary.json
```

- `frames.csv`：每帧原图名称、相对路径、时间点、内部序号、尺寸、mask/overlay 路径、实例数、面积阈值、处理状态。输出文件路径相对于当前序列目录，原图路径相对于 summary 中的 input_dir。
- `instance_tracks.csv`：每个实际检测一行，保存 sequence_id、image_name、frame_index、time_index、instance_id、track_id、质心、面积、形状、轨迹长度和长度标志。x 为列、y 为行，坐标为零起点像素。track_id 仅在所属序列内唯一。
- `tracks.csv`：兼容旧可视化脚本的 frame/label 字段；内容为全部最终轨迹。
- `summary.json`：参数、模型版本、设备、状态、计数、耗时；失败时写错误及当时图像。失败中止整个批次，已完成序列保留；不自动续跑。空检测序列也生成带表头的 CSV 和空白背景 overlay。

## 6. Overlay

从原图作 1%–99% 显示拉伸，叠加半透明分割区域和彩色内轮廓，在质心写最终 tracking ID；同一序列同一 ID 使用稳定颜色。PNG 与原图尺寸一致，`--alpha` 控制填充透明度。overlay 在 gap closing 完成之后生成，保证编号与最终 CSV 一致。当前不画轨迹尾线；密集区域编号可能重叠，可放大查看。

## 7. 与 GT 对齐及后续优化

GT 脚本使用帧内实例 mask 和 instance_tracks.csv，本流程采用相同的关键映射概念，无需转换成 GT XML。先按数据/序列及完整 image_name 对齐，再按 mask IoU 做实例一对一匹配；预测 instance_id/track_id 与 GT 数值不要求相同。通过匹配后的跨帧关联评估身份连续性。GT 无 track_id 的实例仍可用于分割评估。此入口不读取 GT，也尚未自动计算评估指标。

建议用独立验证片段比较两种预处理、固定与自动面积阈值，再调整距离和断轨门限，重点检查小细胞消失、交叉运动、错误断轨连接。自动面积阈值随帧变化可能影响小目标召回。GT 只标注部分帧或部分目标时，需先明确评估范围。后续可增加分割缓存复用、mask IoU 关联及分裂谱系。

## 文件职责

| 文件 | 作用 |
|---|---|
| main.py | 命令行/交互输入、参数校验、启动 |
| pipeline.py | 分组、帧清单、分割与跟踪调度、导出和 overlay |
| segmentation.py | 读图、预处理、模型推理、过滤、整数 TIFF |
| mask_area_filter.py | 面积统计与过滤 |
| tracking_v0.py | 原有匹配、运动预测、断轨连接算法及独立旧入口 |
| tif2png.py | 历史灰度预览工具；新流程不调用，其输出不能作为标签 mask 使用 |
