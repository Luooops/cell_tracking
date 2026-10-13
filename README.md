# Cell tracking

模型代码集中在 `models/`；统一入口只负责参数传递、输入适配和输出，不替换网络、损失、匹配、断轨重连或分裂逻辑。配置使用标准库 JSON，不依赖 Hydra。

## 目录

```text
models/
  gnn_tracking/        GNN、原训练、两种跟踪求解流程、已有权重
  trackastra/          原官方预训练调用与导出流程
  motion_tracking/     MotionAssociation、专用 NPZ 数据准备、训练、窗口推理与解码
  classical_tracking/  当前经典算法、legacy 算法、mitosis 扩展、原分割流水线
utils/                 通用数据准备、读取、评估、入口分发、输入输出适配、可视化
tests/                 motion 模型、数据准备、完整视频推理与统一入口测试
data/<dataset>/        CTC 数据集；scripta_data 为已有 15 孔位数据，未移动或修改
outputs/<dataset>/<model>/<version>/train|test/
outputs/logs/<model>/<version>_<stage>_<dataset>.log   每次运行的完整控制台日志
history_outputs/       按原来源保留历史输出
config.json            数据集、孔位划分与模型参数
```

## 环境

根目录 `requirements.txt` 保留原经典流水线环境约束。GNN 的原依赖记录保留在 `models/gnn_tracking/requirements.txt`；Trackastra 的原版本记录在对应模型 README。它们并非已经验证兼容的统一锁文件，不要把两份冲突的版本约束一起安装。不同环境都可以调用同一入口；模型选择采用延迟导入。

Windows 测试时使用 `$env:MKL_THREADING_LAYER='SEQUENTIAL'`，避免本机 NumPy/绘图库与 PyTorch 的 OpenMP 运行库冲突。

## 使用

从项目根目录执行。相对配置路径以配置文件目录为基准；命令行路径以当前工作目录为基准。`--version` 必填，相同模型/版本/阶段默认不覆盖已有输出；中断后需重跑同一版本时加 `--overwrite`，会先删除该 `<model>/<version>/<train|test>/` 目录再重新运行。

```powershell
python train.py --model gnn_tracking --version v001
python train.py --model gnn_tracking --version cv001 --cv
python train.py --model gnn_tracking --version v002 --gpus 0 1        # 多卡：各 seed（或 --cv 时各折）分到不同 GPU 并行
python test.py --model gnn_tracking --version pretrained --device cpu
python test.py --model gnn_tracking --version v001 --weights outputs/scripta_data/gnn_tracking/v001/train/checkpoints
python test.py --model trackastra --version pretrained
python test.py --model classical_tracking --version baseline
python test.py --model classical_legacy --version baseline
python test.py --model mitosis --version baseline
python test.py --model gnn_division --version experimental
```

默认测试孔保持 `r16c02 r01c20 r07c07`；GNN 默认训练孔和三折划分保持原代码。用 `--wells r01c20` 指定孔位，用 `--set epochs=1` 等覆盖配置中的训练参数。需要训练的入口只有 `gnn_tracking` 和 `motion_tracking`；Trackastra 不增加微调。分裂变体仍属原有实验能力，不代表已用分裂标签训练。

Python 中可用 `from models import get_model`，例如 `get_model('motion_tracking').MotionAssociation(...)`。返回原实现模块，不创建新的算法包装层。

### Motion 完整视频追踪

motion 专用数据准备、增强、网络、训练和推理都在 `models/motion_tracking/`。训练使用 NPZ 窗口；默认推理直接读取整段 CTC 视频，在滑窗中预测关联、合并连接分数并统一解码，输出整段视频的轨迹 ID。

```bash
python -m models.motion_tracking.prepare_npz --output-root outputs/scripta_data/motion_tracking/npz_v2 --max-tokens 2048
python train.py --model motion_tracking --version v2 --train-dir outputs/scripta_data/motion_tracking/npz_v2/train --val-dir outputs/scripta_data/motion_tracking/npz_v2/val
python test.py --model motion_tracking --version v2 --weights outputs/scripta_data/motion_tracking/v2/train/checkpoints/best.pt
```

默认 8 孔训练、4 孔验证、3 孔保留测试。推理输出每孔的 `instance_tracks.csv`、`frames.csv`、`summary.json`、`edges.json`、`association_candidates.csv`、`metrics.json` 和逐帧追踪 PNG。mask 和原图只读，不重新分割。默认 max_gap=3，可跨过 2 帧漏检，不补出缺失 mask，不处理分裂。

默认启用后继辅助损失、10% 保守检测删除和窗口级旋转/镜像增强，可用 `--set` 关闭做消融。旧权重可用于新的视频推理；新增后继头需要重新训练。默认最多 2048 个检测/窗口，若完整候选历史仍超限则报错，不能通过静默缩短丢弃跨漏检候选。

保留 `--windows-dir <NPZ目录>` 作为独立窗口调试模式，它不输出完整视频轨迹。motion 自动从 `<well>_GT/TRA/man_track*.tif` 与 `man_track.txt` 评价，保存每孔 `evaluation/` 下的轨迹明细和 comparison PNG，不再调用 XML 评价器。

已有结果可独立重评，不需要权重或重新推理：

```bash
python evaluate.py --dataset scripta_data --tracking-output outputs/scripta_data/motion_tracking/v2/test --output-root outputs/scripta_data/motion_tracking/v2/evaluate --visualize all
```

这是同一批 CTC 检测上的身份关联评价；不是独立分割评估，也不冒充官方 TRA/SEG。身份未知的 mask 标签 >=50000 不计入身份指标。

研究依据、推理细节、服务器 CLI 和消融计划见 [Motion v2 说明](docs/MOTION_V2.md)。所有结构改进仍须通过真实数据验证，不能把合成测试当作性能证据。

### 多数据集

`data/` 下每个子目录是一个 CTC 数据集：`<序列>/t*.tif` 原图与 `<序列>_GT/TRA/man_track*.tif`、`man_track.txt`。帧号位数不限（`t000`/`t0000`），序列名不限（`r01c20`、`01`）。`config.json` 的 `dataset` 是默认数据集；所有入口（`train.py`、`test.py`、`evaluate.py`、`prepare_npz`）都可用 `--dataset <名称>` 切换，或用 `--data-root` 直接指定目录。

新数据集放进 `data/<名称>/` 即可用 `--dataset <名称>` 测试，序列自动发现，默认测试全部序列。需要训练或固定测试孔时，在 `config.json` 的 `datasets` 中声明：

```json
"my_data": {
  "root": "data/my_data",
  "test_wells": ["01"],
  "train_wells": ["02", "03"],
  "val_wells": ["04"],
  "folds": [["02"], ["03"], ["04"]]
}
```

`test_wells` 永远禁止进入 motion 训练/验证窗口；`folds` 仅 GNN `--cv` 使用，GNN 默认训练孔为 folds 合并，其次 train+val。motion 可把多个数据集的 NPZ 合并训练：`--train-dir <A>/train <B>/train --val-dir <A>/val <B>/val`；NPZ 的 dataset_id 为 `<数据集>/<孔位>`。输出目录为 `outputs/<dataset>/<model>/<version>/<stage>/`，随 `--dataset`（或 `--data-root` 的文件夹名）变化；同一版本可在不同数据集上分别测试，例如用 A 上训练的权重测 B：`python test.py --model motion_tracking --version v2 --dataset B --weights outputs/A/motion_tracking/v2/train/checkpoints/best.pt`。motion 训练只读 NPZ，`--dataset` 仅决定输出文件夹名；合并多数据集训练时可给一个新名称（未在配置中声明也可以）。所有数据集都把 mask 标签 >=50000 视为未知身份。

### 数据与指标

GNN、Trackastra、经典算法读取同一份 `data/<dataset>/<well>` 原图及 `<well>_GT/TRA` mask。Trackastra 的适配器生成原脚本要求的清单，不执行经典跟踪或重新分割。原始 mask 不重标号、不改写；已知身份只用于监督/评估，未知身份 `>=50000` 仍按原规则处理。

统一 CTC 入口采用原 GNN 的 perfect tracks、id switches、purity 定义，保存 `instance_tracks.csv` 和 `metrics.json`。这不等同于旧 XML 位置匹配评估，也不是官方 TRA/CT/IDF1 实现。旧 Trackastra 评估所需代码保留在 `utils/legacy_*`，原入口仍可用：

```powershell
python -m models.trackastra.evaluate --gt <XML或ZIP> --predictions-root <旧格式结果目录> --output-root <评估目录>
python -m models.classical_tracking.main --help
python -m models.motion_tracking.train --help
python -m utils.prepare_data --help
```

数据准备唯一主流程为 `utils/prepare_data.py`，来自原 labcelltracker，算法未修改。其默认参数也未修改；若需重建当前带未知身份检测的数据，必须明确给出 `--keep-unlabelled` 和实际 Cellpose 权重、阈值，不要仅凭默认值声称复现现有数据。

## 验证

```powershell
$env:MKL_THREADING_LAYER='SEQUENTIAL'
python -B -m unittest discover -s tests
```

当前仅保留 motion 相关测试，包括 CTC 评价与 PNG 输出。其他模型测试及历史 AST 指纹校验已移除；历史验证记录见 `docs/REORGANIZATION_REVIEW.md`。
