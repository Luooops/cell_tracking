# 项目结构整理 Review

日期：2026-10-07。

## 结论与边界

本次集中模型目录、修正 import、增加 JSON 参数配置和统一 train/test 调度、适配现有输入输出、归档历史结果。保留原模型结构、forward、loss、训练循环、候选构图、关联求解、gap closing 和分裂变体。没有添加 Trackastra 微调，也没有添加 motion 的窗口融合或自动监督标签生成。

“所有模型直接接收 CTC”目前有一个明确例外：motion 原实现仅支持审核后的 NPZ 窗口。它已接入统一训练/测试入口，但仍使用原 NPZ 契约。若补全 CTC 到训练窗口与完整视频推理，必须另行确定监督和窗口规则，不能作为无算法改动的目录整理暗中完成。

## 目录与接口

| 原位置 | 新位置 | 处理 |
|---|---|---|
| lab_cell_tracker 的网络、图、求解、训练、跟踪 | models/gnn_tracking | 保留算法，改为完整包导入 |
| lab_cell_tracker/data.py | utils/dataset.py | 数据路径默认 data/data；缓存路径独立于模型源代码；读取路径延迟取值 |
| lab_cell_tracker/metrics.py | utils/metrics.py | 计算定义保持原样 |
| lab_cell_tracker/data_preparation/lab_xml_to_ctc_cpsam.py | utils/prepare_data.py | 唯一主数据准备流程，清洗和导出逻辑保持原样 |
| motion_tracking | models/motion_tracking | 原训练、网络、解码全部保留；测试移入 tests |
| trackastra_tracking | models/trackastra | 原预训练关联、导出与旧评估入口保留 |
| main_tracking | models/classical_tracking | 原分割与跟踪流水线保留，公共 mask 工具移出 |
| tracking.py、mitosis_tracker.py | models/classical_tracking/legacy.py、mitosis.py | 两个原实现独立保留，未合并算法 |
| visualize_tracking.py、mask_area_filter.py | utils/visualization.py、mask_processing.py | 公共工具 |

`models.get_model(name)` 延迟返回原实现模块。支持 gnn_tracking、gnn_division、trackastra、motion_tracking、classical_tracking、classical_legacy、mitosis。GNN 分裂变体继续复用已有 GNN 权重，其原有“没有分裂监督”的限制仍存在。

`config.json` 负责默认数据目录、测试孔及各模型参数，标准库读取，未引入 Hydra。命令行支持独立配置文件、数据/输出目录、权重、孔位、设备和参数覆盖。新的运行按 `outputs/<model>/<version>/<train|test>/` 保存参数、权重或预测结果，相同阶段的已有运行拒绝覆盖。

GNN 和 motion 有训练入口。经典算法、分裂规则变体和预训练 Trackastra 不执行伪训练。GNN 原训练轮数、批量、学习率和种子默认不变；motion 原训练参数默认不变。经典主模型的配置来自原 main_tracking/main.py 和 pipeline.py 的组合参数。

预置的三份 GNN 权重作为模型资产保留在 models/gnn_tracking/weights，新训练的权重进入 outputs。旧模型模块仍可通过 `python -m models.<...>` 调用原入口；根目录 README 是当前用法来源。

## 输入输出适配

- 数据没有迁移、重分割或改写。真实根目录是 data/data，共 15 个孔位，每孔 30 对图像/mask；检查了预期文件配对。
- GNN 使用原检测表和 crop 提取；读取路径从新配置传入。
- Trackastra 使用 CTC 图像和 mask 生成它原来要求的 frames.csv/instance_tracks.csv 清单，随后调用原 track_sequence。输入清单保存在该次运行的 test/inputs，便于追溯。不先运行经典跟踪。
- 经典算法逐帧使用原检测提取和 tracker.update，主模型参数与原 pipeline 相同；若启用 gap closing，调用原函数。分裂变体保留母子关系导出。
- motion 训练直接调用原 main；测试从原 checkpoint 恢复网络和特征标准化参数，逐窗口调用原 forward 和 decode_tracks，不合并窗口、不静默截断。
- CTC 统一指标使用原 GNN score 定义，未知身份检测按原规则忽略；不是官方 TRA/CT/IDF1，也不与旧 XML 匹配指标混称。

## 删除与历史归档

删除旧 gt_process 生成、检查和修正脚本；删除独立 mask_eval.py、png2gif.py、tif2png.py；根目录单图分割演示 test.py 替换为统一测试入口。删除旧 perfect_tracks.py、export_gt_reports.py 及已失效的旧评估说明、过时平台教程。

原 track_eval/evaluate.py 和 metrics.py 仍是已有 Trackastra 评估与测试的依赖，迁为 utils/legacy_evaluate.py、legacy_metrics.py；从旧 GT 工具中只提取它们仍需的两个读取函数到 legacy_io.py。这样去掉了旧 GT 处理流程，同时没有破坏原 Trackastra 评估流程。

11 个历史输出目录归入 history_outputs，保留 gt_process/main_tracking/track_eval 来源层级。根目录两份 GIF 也已归档，内容哈希与整理前一致。对可定位的历史输出路径引用做了更新；外部原图路径保持原记录，未宣称这些外部路径在当前机器均有效。迁移映射记录在 history_outputs/migration.json。

## 实际验证

1. `python -B -m unittest discover -s tests`：18 项测试通过。包括原 motion 的梯度、解码、微型拟合、权重读回和泄漏防护；原 Trackastra 导出/评估回归；原指标回归；统一入口的经典变体、模拟 Trackastra、motion 一轮训练到窗口推理和防覆盖检查。
2. 17 个文件中的 110 个原函数/类定义通过 AST 指纹比较；仅忽略 import、文档字符串及 Python 版本的空泛型字段差异。网络、图构造、求解、训练函数及数据准备主体在覆盖范围内。CLI 与少数路径接入函数不在此指纹范围，不将此检查等同于完整运行等价证明。
3. 42 个 Python 文件语法解析通过。
4. 三份 GNN 权重逐一与 Git HEAD 对应文件哈希一致。
5. 真实数据 r01c20 的 30 帧通过统一 classical_tracking 入口跑通。输出保留在 outputs/classical_tracking/structure_review/test，作为可复查验证产物。所得 perfect=19.0141%、id_switches=386、purity=0.942801；此数值只是本次运行记录，不是性能改进声明或与旧 XML 指标的比较。

## 验证限制与后续使用

- 当前 torch 环境没有 torch_geometric 或 trackastra，未执行 GNN 真实权重推理/训练，也未下载或运行真正的 Trackastra 网络；Trackastra 的接口验证使用模拟网络返回图。未安装或改动用户环境依赖。
- 未执行完整 GNN/motion 重新训练。motion 的微型训练用于接口回归，不构成真实数据性能验证。
- 保留原两套依赖记录，未凭猜测合并为一个兼容锁文件。README 说明不同环境可调用同一入口。
- Windows 测试需在启动前设置 MKL_THREADING_LAYER=SEQUENTIAL。本机默认沙箱临时目录曾拒绝写入，回归测试在获准的沙箱外执行后完成。
- 没有新增 git commit，也没有修改数据集。迁移用临时脚本已清理，保留有持续价值的回归测试、结构指纹和真实验证结果。
