# Cell tracking

模型代码集中在 `models/`；统一入口只负责参数传递、输入适配和输出，不替换网络、损失、匹配、断轨重连或分裂逻辑。配置使用标准库 JSON，不依赖 Hydra。

## 目录

```text
models/
  gnn_tracking/        GNN、原训练、两种跟踪求解流程、已有权重
  trackastra/          原官方预训练调用与导出流程
  motion_tracking/     原 MotionAssociation、NPZ 窗口训练、解码
  classical_tracking/  当前经典算法、legacy 算法、mitosis 扩展、原分割流水线
utils/                 数据准备、读取、评估、输入输出适配、可视化
tests/                 原回归测试、入口测试、算法结构保留校验
data/data/             已有 15 孔位数据；未移动或修改
outputs/<model>/<version>/train|test/
history_outputs/       按原来源保留历史输出
config.json            数据路径、孔位与模型参数
```

## 环境

根目录 `requirements.txt` 保留原经典流水线环境约束。GNN 的原依赖记录保留在 `models/gnn_tracking/requirements.txt`；Trackastra 的原版本记录在对应模型 README。它们并非已经验证兼容的统一锁文件，不要把两份冲突的版本约束一起安装。不同环境都可以调用同一入口；模型选择采用延迟导入。

Windows 测试时使用 `$env:MKL_THREADING_LAYER='SEQUENTIAL'`，避免本机 NumPy/绘图库与 PyTorch 的 OpenMP 运行库冲突。

## 使用

从项目根目录执行。相对配置路径以配置文件目录为基准；命令行路径以当前工作目录为基准。`--version` 必填，相同模型/版本/阶段不覆盖已有输出。

```powershell
python train.py --model gnn_tracking --version v001
python train.py --model gnn_tracking --version cv001 --cv
python test.py --model gnn_tracking --version pretrained --device cpu
python test.py --model gnn_tracking --version v001 --weights outputs/gnn_tracking/v001/train/checkpoints
python test.py --model trackastra --version pretrained
python test.py --model classical_tracking --version baseline
python test.py --model classical_legacy --version baseline
python test.py --model mitosis --version baseline
python test.py --model gnn_division --version experimental
```

默认测试孔保持 `r16c02 r01c20 r07c07`；GNN 默认训练孔和三折划分保持原代码。用 `--wells r01c20` 指定孔位，用 `--set epochs=1` 等覆盖配置中的训练参数。需要训练的入口只有 `gnn_tracking` 和 `motion_tracking`；Trackastra 不增加微调。分裂变体仍属原有实验能力，不代表已用分裂标签训练。

Python 中可用 `from models import get_model`，例如 `get_model('motion_tracking').MotionAssociation(...)`。返回原实现模块，不创建新的算法包装层。

### Motion 的原有边界

该模型的数据契约是 NPZ 窗口。现在可使用独立的数据适配脚本从现有实验室 CTC 数据生成窗口，网络和训练循环不变：

```powershell
python -m utils.prepare_motion_npz --output-root outputs/motion_tracking/npz_v001
python train.py --model motion_tracking --version v001 --train-dir outputs/motion_tracking/npz_v001/train --val-dir outputs/motion_tracking/npz_v001/val
```

默认使用 8 个开发孔训练、4 个开发孔验证，保留原 3 个测试孔。分组单位明确为孔位，验证同实验内的跨孔泛化，不宣称独立实验泛化。默认最多 6 帧/窗口、1024 个检测；超限时缩短到完整的连续帧，若两帧仍超限则报错。每个窗口保留全部检测，不能确定的前驱监督设为 -2。详细标签规则、归一化和生成记录见 `docs/MOTION_NPZ.md` 与输出目录的 `manifest.json`。

也可以提供自行审核的兼容窗口：

```powershell
python train.py --model motion_tracking --version v001 --train-dir <NPZ训练目录> --val-dir <NPZ验证目录>
python test.py --model motion_tracking --version v001 --windows-dir <NPZ测试目录> --weights outputs/motion_tracking/v001/train/checkpoints/best.pt
```

每个窗口保持独立，测试复用权重中的特征均值/标准差，调用原 forward 与 decode_tracks；输出逐窗口 CSV 和边列表，不宣称完整视频评估。不自动拆分超限窗口。训练仍保留原 dataset_id 泄漏检查、未知监督和候选门限检查。

### 数据与指标

GNN、Trackastra、经典算法读取同一份 `data/data/<well>` 原图及 `<well>_GT/TRA` mask。Trackastra 的适配器生成原脚本要求的清单，不执行经典跟踪或重新分割。原始 mask 不重标号、不改写；已知身份只用于监督/评估，未知身份 `>=50000` 仍按原规则处理。

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

`test_preservation` 对照整理前 Git HEAD 的函数/类 AST 指纹，忽略 import 和文档字符串，验证核心算法未改写。它不替代真实模型数值验证；详细实测范围见 `docs/REORGANIZATION_REVIEW.md`。
