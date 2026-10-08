> Historical model documentation. For current commands and paths, use the root README.md.

# Trackastra 基线

用 [Trackastra](https://github.com/weigertlab/trackastra)（Gallusser & Weigert, ECCV 2024）的**官方预训练模型**，
对 `main_tracking` 已经生成的 mask 重新做关联（tracking）。分割结果完全不变，只替换"哪些检测属于同一个细胞"这一步，
输出格式和 `main_tracking` 一样，所以可以直接用 `track_eval/evaluate.py` 评估，和经典算法在同一批 mask、同一套指标下比较。

注意：这里**没有训练**。Trackastra 没有用我们的数据训练或微调，用的是它发布的权重。

## 安装

在已有 torch 的环境里：

```powershell
pip install trackastra
```

第一次运行会自动下载模型权重（约几十 MB，保存到用户目录下的 `trackastra/models`）。
测试过的版本：trackastra 0.5.6，torch 2.6.0+cu124，numpy 2.2.6，Python 3.10。有 GPU 时自动用 GPU，`--cpu` 强制用 CPU。

## 运行

从项目根目录运行。先用 `models/classical_tracking/main.py` 跑出分割和经典 tracking，然后：

```powershell
# 处理 models/classical_tracking/outputs 下的所有序列（原图目录取每个序列 summary.json 里记录的 input_dir）
python models/trackastra/run_trackastra.py --predictions-root models/classical_tracking/outputs

# 或只处理一个序列，并显式指定原图目录
python models/trackastra/run_trackastra.py `
  --sequence-dir "models/classical_tracking/outputs/<UUID>__r16c13/f01__p01__ch02" `
  --input-dir "E:\aws_gt_data\<UUID>\r16c13"
```

评估（和经典算法的评估命令相同，只是换了 `--predictions-root`）：

```powershell
python track_eval/evaluate.py --gt "<XML 或 ZIP>" --predictions-root models/trackastra/outputs
```

主要参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | `general_2d` | 预训练模型，可选 `general_2d` 或 `ctc` |
| `--mode` | `greedy_nodiv` | 连接方式：`greedy_nodiv`（不处理分裂）、`greedy`（允许分裂）、`ilp`（整数线性规划，允许分裂） |
| `--output-root` | `models/trackastra/outputs` | 输出根目录 |
| `--copy-masks` | 关 | 把 mask 也复制到输出目录（默认不复制，`frames.csv` 里的 `mask_path` 指回原来的 mask） |
| `--overwrite` | 关 | 输出目录非空时覆盖 |

## 输出

```text
models/trackastra/outputs/<数据UUID>__<孔位>/f01__p01__ch02/
  frames.csv            和 main_tracking 相同（mask_path 指向所用的 mask，overlay_path 为空）
  instance_tracks.csv   每个检测一行，列和 main_tracking 相同；track_id / track_length / passes_min_track_length 是 Trackastra 的结果
  tracks.csv            兼容旧可视化脚本
  summary.json          status、Trackastra 模型和模式、软件版本、原分割参数、计数、耗时
```

检测（instance_id、质心、面积、形状）原样沿用 `main_tracking` 的 `instance_tracks.csv`，只改 `track_id`。
`track_id` 从 1 开始，仅在序列内唯一。没有被 Trackastra 连到任何轨迹的检测，会各自成为一条单帧轨迹。
本脚本不生成 overlay。

## 做法

1. 按 `frames.csv` 的顺序读入一个序列的全部 mask 和对应原图。
2. Trackastra 的 transformer 对相邻帧之间的每一对检测打一个关联分数（0–1）。
3. 连接：`greedy_nodiv` 只保留分数 ≥ 0.5 的配对，按分数从高到低逐个接受，每个检测最多一个前驱、一个后继。
   没有合格候选的检测就不连（不存在"必须配上一个"的约束）。
4. 从它的解图读出每条轨迹，写回 `instance_tracks.csv`。

## 结果

### 本仓库自带的示例序列（r16c13，`main_tracking` 的 cpsam_v2 mask，`track_eval/evaluate.py`，max-distance 45）

| 指标 | 经典算法（仓库中的现有输出） | Trackastra（general_2d，greedy_nodiv） |
|---|---|---|
| GT 轨迹完整跟踪成功率 | 11.51%（51/443） | 21.90%（97/443） |
| 条件身份保持率 | 82.49% | 97.76% |
| GT 相邻连接成功率（含歧义） | 71.01% | 84.15% |
| GT 相邻连接成功率（排除歧义） | 72.36% | 85.75% |
| 预测混用比例 | 22.03%（274/1244） | 18.67%（126/675） |
| 预测轨迹数 | 1286 | 701 |

位置覆盖率两者相同（89.99%），因为检测是同一批。GT 用的是 2026-10 重新标注的版本。30 帧序列在 GPU 上约 7–9 秒。

同一序列，用 `track_eval/perfect_tracks.py`（完美轨迹比例和 ID 切换，定义见 `track_eval/README_perfect_tracks.md`）：

| 指标 | 经典算法（仓库中的现有输出） | Trackastra（general_2d，greedy_nodiv） |
|---|---|---|
| 完美轨迹比例 | 19.28%（91/472） | 38.77%（183/472） |
| ID 切换 | 1248 | 377 |
| 纯度 | 0.924 | 0.940 |

```powershell
python track_eval/perfect_tracks.py --gt "<XML 或 ZIP>" --predictions-root models/trackastra/outputs
```

### 为什么默认用 general_2d + greedy_nodiv

在 15 个重新标注的孔上试了 2 个预训练模型 × 3 种连接方式（检测为实验室微调的 Cellpose，指标为完美轨迹比例，
即 `track_eval/perfect_tracks.py` 的口径）：

| 模型 | 模式 | 3 个留出孔 | 其余 12 个孔 |
|---|---|---|---|
| general_2d | greedy_nodiv | **49.0%** | **52.3%** |
| general_2d | greedy | 45.7% | 49.8% |
| general_2d | ilp | 44.5% | 47.5% |
| ctc | greedy_nodiv | 46.7% | 48.4% |
| ctc | greedy | 41.5% | 43.9% |
| ctc | ilp | 36.3% | 38.7% |

允许分裂的模式在我们的数据上更差：这个细胞系不分裂，模型偶尔把普通细胞判成分裂，轨迹就被拆开了。

## 已知限制

- **不能跨过漏检帧。** 公开版本的预测只包含相邻帧的配对分数。即使把连接步骤的 `delta_t` 调大，候选图里也不会出现
  跨帧的边（实测 `delta_t` = 1–4 结果完全相同）。某个细胞在某一帧被 Cellpose 漏检后，前后两段会成为两条轨迹。
  经典算法有 `max_lost` 和 gap closing，这一点上两者不同。
- **没有用我们的数据训练。** 用实验室数据微调后可能更好，这里没有做。
- **结果取决于输入的 mask。** 换分割参数或分割模型后需要重新跑。
- `greedy` 和 `ilp` 模式下，分裂后的子细胞会得到新的 `track_id`（CTC 约定），当前输出不记录母子关系。
