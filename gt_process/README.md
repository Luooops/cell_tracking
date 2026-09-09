# GT mask 与 overlay

在项目根目录运行（依赖 `numpy pillow tifffile`）：

```powershell
python gt_process/generate_gt_masks.py --gt "E:\aws_gt_data\4cbffe68-5969-4f0a-8be6-9e02efd9720f\r16c13" --data-dir "E:\aws_gt_data\4cbffe68-5969-4f0a-8be6-9e02efd9720f\r16c13"
```

- `--gt`：普通 XML、内部含一个 XML 的 ZIP（包括扩展名为 `.xml` 的 ZIP），或仅含一个 XML/ZIP 的目录。
- `--data-dir`：原图目录，递归搜索。严格按 XML 的图像文件名匹配，不会自动切换通道；缺失或同名多份会报错。
- `--output-root`：默认是脚本旁的 `outputs`。子目录名从原图路径提取数据 UUID、孔位编号和末级目录名。已有非空结果目录会报错，重新运行时可指定另一输出根目录。
- `--alpha 0.35`：overlay 填充透明度；`--no-ids` 隐藏文字；`--label cell` 选择类别。

结果结构：

```text
outputs/<数据UUID>__<孔位>/
  masks/<原图文件名去扩展名>_mask.tiff
  overlays/<原图文件名去扩展名>_overlay.png
  instance_tracks.csv
  summary.json
```

mask 为单通道整数 TIFF：背景 0，细胞为帧内实例编号 1、2、3……，不是 track ID。通常保存为 uint16，超过 65535 个实例时使用 uint32。CSV 保存原始 track ID（空值保留）、实例编号、图像名称和 XML 帧编号，并记录栅格化面积及最终 mask 面积。

多边形顶点按 `floor(x+0.5)` 取整，使用 Pillow 填充并在图像边界裁剪。重叠像素归 XML 中后出现的多边形；summary 记录每帧重叠像素数。完全被覆盖的实例在 CSV 中的 mask_pixels 为 0。

Overlay 对原图作 1%–99% 分位数显示拉伸，不改变 mask；相同非空 track ID 使用固定颜色。文字是 track ID；空 ID 显示 `?实例编号`。支持单帧灰度/RGB/RGBA 图像，不自动选择多页 TIFF 的页或三维图像的切片。
