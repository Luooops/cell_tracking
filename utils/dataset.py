"""
data.py
=======
Step 0 of the pipeline: segmentation masks + raw images -> a flat detection table and one image crop per cell.

Input layout (Cell Tracking Challenge style; how it was made is described in the README and in
data_preparation/):

    <LAB_ROOT>/<well>/t000.tif ...                      raw nuclear-channel frames (2160 x 2160)
    <LAB_ROOT>/<well>_GT/TRA/man_track000.tif ...       label masks: pixel value = id of the cell

The masks are the fine-tuned Cellpose detections (cellprob -2, flow 0.8). The pixel value is the GT track id where
the annotation could be matched to a detection, and a unique value >= UNLABELLED_OFFSET (50000) where it could not:
such a cell is a real detection, but nobody knows which track it belongs to.

Output of load_well(well):

    df      one row per detected cell, sorted by (frame, track_id), index 0..n-1
            columns: video, frame, track_id, x, y, area, mean_intensity
    crops   float16 [n, 64, 64], row i = the appearance crop of the cell in df row i

How a crop is made:
    1. cut the bounding box of the cell out of the raw frame;
    2. keep the cell's own pixels only -- everything outside its mask becomes 0 (a quarter of a bounding box is
       background and 3% belongs to neighbouring cells; neither says anything about this cell);
    3. scale the cell's own pixels to [0, 1] with their 1% / 99% quantiles (robust to single hot pixels);
    4. pad the box to a square with zeros (the shape is not stretched), then resize to 64 x 64.
Masking the crop and padding instead of stretching is also what CellTrack-GNN (Ben-Haim & Riklin-Raviv, ECCV 2022)
and DeepCell's 'fixed' crop mode do.

The only things the tracker reads are x, y, area, mean_intensity and the crop. track_id is used for the training
labels and for scoring, never as a model input.

Reading 2160 x 2160 frames is slow, so every well is cached in <CACHE>/<well>.pt on first use. The mirrored
versions used for training are derived from the cache: mirroring the image mirrors the centroid coordinate and the
crop, and changes nothing else.

    python data.py                 # build the cache for all 15 wells
    python data.py r16c02          # one well, and print what it contains
"""
import os, re, sys, glob
import numpy as np, pandas as pd, torch, tifffile
from skimage.measure import regionprops
from skimage.transform import resize

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "gnn_tracking")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB_ROOT = os.environ.get("LAB_TRACKER_DATA", os.path.join(PROJECT_ROOT, "data", "data"))
CACHE = os.environ.get("LAB_TRACKER_CACHE", os.path.join(PROJECT_ROOT, "outputs", "cache", "gnn_tracking"))

CROP_SIZE = 64
UNLABELLED_OFFSET = 50000          # track_id >= this: a detection without a ground-truth identity

# 15 annotated wells. The 3 test wells are for final evaluations only; every design decision is made on the
# 12 development wells with the 3-fold split below (whole wells held out, never single frames).
TEST_WELLS = ["r16c02", "r01c20", "r07c07"]
FOLDS = [["r08c13", "r11c12", "r03c12", "r14c13"],
         ["r15c12", "r12c13", "r07c02", "r09c01"],
         ["r13c12", "r16c13", "r09c09", "r14c03"]]
DEV_WELLS = [w for fold in FOLDS for w in fold]


def appearance_crop(img, own, size=CROP_SIZE):
    """img: the bounding box cut from the raw frame; own: bool mask of the cell inside that box. -> float16 [size, size]"""
    lo, hi = np.quantile(img[own], [0.01, 0.99])
    c = np.where(own, np.clip((img - lo) / (hi - lo + 1e-6), 0, 1), 0.0)
    h, w = c.shape
    s = max(h, w)
    square = np.zeros((s, s), np.float32)
    square[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = c
    return np.clip(resize(square, (size, size), anti_aliasing=True), 0, 1).astype(np.float16)


def read_well(well, root=None):
    """Masks + images of one well -> (df, crops, image shape)."""
    root = LAB_ROOT if root is None else root
    files = sorted(glob.glob(os.path.join(root, f"{well}_GT", "TRA", "man_track*.tif")))
    if not files:
        raise FileNotFoundError(f"no masks for {well} under {root}")
    rows, crops = [], []
    for f in files:
        frame = int(re.search(r"man_track(\d+)", os.path.basename(f)).group(1))
        mask = tifffile.imread(f)
        img = tifffile.imread(os.path.join(root, well, f"t{frame:03d}.tif")).astype(np.float32)
        for p in regionprops(mask, intensity_image=img):
            cy, cx = p.centroid
            r0, c0, r1, c1 = p.bbox
            rows.append(dict(video=well, frame=frame, track_id=int(p.label), x=float(cx), y=float(cy),
                             area=int(p.area), mean_intensity=float(p.mean_intensity)))
            crops.append(appearance_crop(img[r0:r1, c0:c1], p.image))
    raw = pd.DataFrame(rows)
    order = raw.sort_values(["frame", "track_id"]).index.to_numpy()
    return raw.iloc[order].reset_index(drop=True), np.stack(crops)[order], mask.shape


def load_well(well, flip=None):
    """Cached read_well. flip = None | 'h' | 'v' returns the well as if image and masks had been mirrored
    left-right / up-down before anything was measured (training augmentation)."""
    path = os.path.join(CACHE, f"{well}.pt")
    if os.path.exists(path):
        d = torch.load(path, weights_only=False)
        df, crops, (height, width) = d["df"], d["crops"].numpy(), d["shape"]
    else:
        df, crops, (height, width) = read_well(well)
        os.makedirs(CACHE, exist_ok=True)
        torch.save({"df": df, "crops": torch.from_numpy(crops), "shape": (height, width)}, path)
    if flip == "h":                                 # column c of the image becomes column width-1-c
        df = df.assign(x=width - 1 - df.x); crops = crops[:, :, ::-1]
    elif flip == "v":
        df = df.assign(y=height - 1 - df.y); crops = crops[:, ::-1, :]
    return df, np.ascontiguousarray(crops)


if __name__ == "__main__":
    wells = sys.argv[1:] or DEV_WELLS + TEST_WELLS
    for w in wells:
        df, crops = load_well(w)
        n_unl = int((df.track_id >= UNLABELLED_OFFSET).sum())
        print(f"{w}: {df.frame.nunique()} frames, {len(df)} detections ({len(df) / df.frame.nunique():.0f} per frame), "
              f"{df.track_id[df.track_id < UNLABELLED_OFFSET].nunique()} GT tracks, {n_unl} detections without identity, "
              f"crops {crops.shape} {crops.dtype}")
    if len(wells) == 1:
        print(df.head(8).to_string())
