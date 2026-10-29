# CellTrack-GNN

Ben-Haim & Riklin-Raviv, ECCV 2022（github.com/talbenha/cell-tracker-gnn）。适配器 `run_celltrack_gnn.py` 在官方环境里原样调用三个官方推理脚本，结果转成本项目格式（`instance_tracks.csv`、`lineage.json`、`metrics.json`），再用 `evaluate_ctc` 评估，与其他模型一致。

```bash
python test.py --model celltrack_gnn --dataset scripta_data_refine2 --version seed0 \
    --weights models/celltrack_gnn/weights/seed0 --set repo='"<cell-tracker-gnn 仓库>"' --set python='"<ctgnn 环境>/bin/python"'
python -m models.gnn_tracking.evaluate_ctc --dataset scripta_data_refine2 \
    --run outputs/scripta_data_refine2/celltrack_gnn/seed0/test --output-root <评估目录>
```

- 权重：`weights/seed0`、`weights/seed1`（两个训练 seed，config 默认 seed0；不进 git，与代码一起拷到服务器）。每个是一个模型文件夹：`metric_learning/all_params.pth` + `gnn/checkpoints/<唯一>.ckpt` + `gnn/.hydra/config.yaml`（`inference_clean.py` 从 checkpoint 上两级读取配置）。
- 每孔的官方输入与中间结果在 `<输出>/work/<孔>/`：`01/`（原图硬链接）、`01_SEG_RES/`（检测 mask 的副本，预处理会删去小于 `min_cell_size` 的碎片并原地改写）、`01_CSV/`、`01_RES_inference/`、`01_RES/`。`data/` 只读。
- 转换：每个检测取覆盖其像素最多的结果编号为 `track_id`；被官方清理掉的检测，以及同一帧与更大检测共用一个结果编号的检测，记为单检测轨迹（`summary.json` 的 `removed_by_cleanup` / `displaced`）。共用编号来自官方后处理：它不要求一对一，两个细胞连到同一个后继时，两条轨迹整段被赋成同一个编号。这个规则对该方法偏严，被并入的轨迹在合并前的连接也会丢失；结果和说明见 `development/RESULTS.md`。`res_track.txt` 的父轨迹（官方后处理把在第 t 帧开始的轨迹接到第 t−1 帧结束、距离最近的轨迹）原样写入 `lineage.json`：身份指标只看 `track_id`，CTC 评估也计入父轨迹。
- 训练用官方代码与配置完成；训练记录（日志、完整配置、测试分数，不含权重）在 `development/stage5_celltrack_gnn/training/`：12 口开发孔训练，测试孔 r16c02 r01c20 r07c07。
