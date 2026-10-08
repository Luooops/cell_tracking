> Historical model documentation. For current commands and paths, use the root README.md.

# 统一评估

`evaluate.py` 直接调用项目的 `track_eval/evaluate.py`，使用相同的位置匹配、
默认门限（priority-distance=30、max-distance=45）、指标、CSV、JSON 和可视化。
默认读取 `models/trackastra/outputs`，输出到 `models/trackastra/evaluation_outputs`。
支持共享评估器的全部参数，包括 `--tracking-output`、`--output-root`、`--batch`、
`--visualize` 和 `--allow-missing`。

从项目根目录运行：

```powershell
& D:\MiniConda\envs\torch\python.exe .\trackastra_tracking\evaluate.py `
  --gt "E:\aws_gt_data_new" --batch
```

批量输出：`evaluation_outputs/<UUID>/<well>/`。单个 GT 输出：
`evaluation_outputs/<UUID>__<well>__<时间戳>/`。
已有预测可直接评估，无需重新推理；已有结果不会自动改写。
比较历史实验时，应显式使用相同的位置门限和其他评估参数。

本次导出修正：拒绝未完成或时间不连续的源序列；支持空检测 CSV；
校验 CSV 与 mask 实例一致；统一两个 CSV 的 mask_path；支持复制外部路径的 mask；
保留 min_track_length；防止覆盖源序列或输出到根目录之外。

验证：

```powershell
& D:\MiniConda\envs\torch\python.exe -B -m unittest trackastra_tracking.test_tracking track_eval.test_metrics
```

测试使用模拟关联图，验证导出、空序列、缺帧拒绝和共享评估结果一致性，
不代表真实预训练模型的精度验证。
