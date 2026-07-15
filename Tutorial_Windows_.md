# Cellpose GUI:标注 Ground Truth 与模型微调(Windows)

一份指南,包含 **(1)** 用 Cellpose GUI 制作 ground-truth(GT)实例 mask,以及
**(2)** 用 human-in-the-loop 流程微调 Cellpose 模型。

> 基于 **Cellpose v4.2.0**, 在 Windows 11 + NVIDIA GPU 上测试通过。

---

## 参考环境

> 以下是我自己使用的版本(Windows + NVIDIA GPU)。

- **Python** 3.10.20

| 包 | 版本 | 说明 |
|---|---|---|
| cellpose | 4.2.0 | 主程序 |
| torch | 2.6.0+cu124 | GPU(CUDA 12.4)版;无 GPU/Mac 会是 `2.6.0`(CPU 版) |
| torchvision | 0.21.0+cu124 | |
| numpy | 2.2.6 | |
| scipy | 1.15.3 | |
| scikit-image | 0.25.2 | |
| tifffile | 2025.5.10 | 读写 TIFF |
| imagecodecs | 2025.3.30 | |
| matplotlib | 3.10.9 | |
| pandas | 2.3.3 | |
| PyQt6 | 6.11.0 | GUI 界面 |
| pyqtgraph | 0.14.0 | GUI 显示 |
| QtPy | 2.4.3 | |
| superqt | 0.8.2 | |
| natsort | 8.4.0 | |

> 只要 `pip install "cellpose[gui]"` 装的是 **cellpose 4.2.x**,以上依赖会自动装成
> 兼容版本,一般不用逐个指定。

## Part 0 — 环境配置

### 1. 安装 Miniconda 
从 <https://docs.conda.io/en/latest/miniconda.html> 下载 **Windows 64-bit** 安装包,
运行(一路默认)。

### 2. 打开终端
打开 **Anaconda Prompt**。以下所有命令都在这里输入。

### 3. 创建环境并安装 Cellpose(带 GUI)
```bash
conda create -n cellpose python=3.10 -y
conda activate cellpose
pip install "cellpose[gui]"
```
`[gui]` 会装图形界面(PyQt、pyqtgraph)。pip 同时会装 PyTorch;Windows 上有 NVIDIA GPU
会自动用 CUDA(没有 GPU 也行,只是用 CPU 慢一点)。

### 4. 验证是否安装成功
```bash
python -c "import cellpose, torch; print(cellpose.__version__, 'cuda:', torch.cuda.is_available())"
```

---

## Part 1 — 用 Cellpose GUI 制作 Ground Truth

### 第 1 步:把图片转成 RGB
我们的显微镜图片是单通道灰度图。Cellpose v4.2.0 的 GUI 读单通道 TIFF 有个 bug
(报 `channel_axis is not None` 错误)。解决办法:把灰度图复制成 3 个相同的通道(RGB)
——像素内容完全不变。

可以直接复制并运行下面的 `convert_to_rgb.py`, 脚本:

```python
import os
import glob
import numpy as np
import tifffile

# ---- 改这两个路径 ----
IN_DIR = r"C:\存放灰度图的文件夹"      
OUT_DIR = r"C:\转换后RGB的输出文件夹"   

os.makedirs(OUT_DIR, exist_ok=True)
for p in glob.glob(os.path.join(IN_DIR, "*.tif*")):
    a = np.squeeze(tifffile.imread(p))          # (H, W) 灰度
    rgb = np.stack([a, a, a], axis=-1)          # (H, W, 3) RGB
    stem = os.path.splitext(os.path.basename(p))[0]
    tifffile.imwrite(os.path.join(OUT_DIR, stem + "_rgb.tiff"), rgb)
    print("已写入", stem + "_rgb.tiff")
```

### 第 2 步:启动 GUI
```bash
conda activate cellpose
python -m cellpose
```
会弹出一个 Cellpose 窗口。

### 第 3 步:加载图片
把 `_rgb.tiff` 文件手动拖进窗口 (或 `File → Load image`)。

### 第 4 步:跑模型得到初始 mask
在 **Segmentation** 面板，模型保持 **`cpsam_v2`**，点 **`run`**。几秒后它会在细胞上画出
彩色 mask。这样我们只需要改错，不用从零画。

> 建议: 先关掉自动保存 —— `File → "Disable autosave _seg.npy file"` —— 免得重跑模型把改好的东西覆盖了。

### 第 5 步:修改 mask
| 操作 | 方法 |
|---|---|
| **补一个漏掉的细胞** | **右键按住拖**，沿细胞画一圈,回到起点自动闭合成新 mask(`single stroke` 要勾上)。 |
| **删一个错的细胞** | **Ctrl + 左键**点它。 |
| **删多个** | 用 *delete multiple ROIs → click-select / region-select → done*。 |
| **显示/隐藏 mask** | 按 **X** |
| **显示/隐藏轮廓** | 按 **Z**(方便检查粘连细胞有没有分开) |
| **放大** | **鼠标滚轮** |

完整快捷键:**Help 菜单 → key/mouse commands**。

### 第 6 步:保存
按 **`Ctrl + S`**。会在**原图旁边**生成一个 `<图片名>_seg.npy` 文件，里面就是修正后的
实例 mask —— 这就是这一帧的 ground truth。

每标一帧,重复第 3–6 步。

---

## Part 2 — 用 Human-in-the-Loop 微调模型

不用从零标所有图，而是让模型先预测，我们只改它的错，重训，模型每一轮都变好 ——
于是每张新图都改得更快。

### 循环流程
```
1. 加载一张【新图】(模型没训练过的)
2. 跑当前模型  →  它给出预测 mask
3. 改它的错     (Ctrl+左键=删, 右键拖=加)
4. Ctrl+S 存修正后的 _seg.npy(存在图片旁边)
5. 攒几张后,重新训练(见下)
6. 新的(更准的)模型去预测下一批图
7. 重复 —— 每轮改得越来越少
```

### 怎么训练新模型
1. 把要训练的图片(每张带它的 `_seg.npy`)**放在同一个文件夹**里。
2. GUI 里:**`Models` 菜单(左上角)→ `Train new model`**。
3. 对话框里:
   - **initial model**:`cpsam_v2`
   - **learning_rate**:`1e-5`,**weight_decay**:`0.1`,**n_epochs**:`100`(默认即可)
   - 右边那列会列出图片和 **`# of masks`** —— 这就证明它找到了你的 `_seg.npy`。
     训练用的是文件夹里**"图片 + `_seg.npy`"的配对**。
4. 点 **OK**。开始训练(进度在启动 GUI 的那个终端里滚动)。
   - GUI 窗口可能显示 "(未响应)" —— 这是正常的，因为训练占住了主线程，不要关闭窗口，
     训完自己会恢复。可以在终端实时查看训练进度。
5. 训好的模型出现在 **`user-trained models`** 面板里。选它、点 **`run`** 就能用在下一批图上。

### 对于一轮喂多少张图片给模型，以下是我的想法
- **早期(模型还弱)**:一轮 **1–3 张**，然后重训。模型早期进步最快，勤重训能让下一批更好改;
- **后期(模型变强)**:一轮 **5–10 张**(这时改得很快了);
- **训练集是累加的** —— 每轮把新改好的图加进文件夹,重训时用**全部**。集合越滚越大、
  模型越来越准、你改得越来越少;
- 不要一次把所有图标完然后一次性训练 —— 因为随着模型越来越准，理论上我们手动要改的东西会越来越少，所以我们选择多次训练。

### 留一个干净的测试集
我们单独拿出几帧(比如 3–5 张)放在另一个文件夹，不拿去训练。

---

## 速查

```bash
# 配置(一次性)
conda create -n cellpose python=3.10 -y
conda activate cellpose
pip install "cellpose[gui]"

# 每次用
conda activate cellpose
python -m cellpose
```

| 按键 | 操作 |
|---|---|
| 右键 + 拖 | 画 / 加一个细胞 |
| Ctrl + 左键 | 删一个细胞 |
| X | 显示/隐藏 mask |
| Z | 显示/隐藏轮廓 |
| 滚轮 | 缩放 |
| Ctrl + S | 保存 `_seg.npy` |
