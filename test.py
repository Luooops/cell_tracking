import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from cellpose import models


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(img, p_low=1, p_high=99, bg_sigma=80, black_gamma=1.8):
    img = img.astype(np.float32)

    # 1. robust normalization
    lo, hi = np.percentile(img, (p_low, p_high))
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)

    # 2. background subtraction
    bg = gaussian_filter(img, sigma=bg_sigma)
    img = img - bg
    img = np.clip(img, 0, 1)

    # 3. darken background
    # img = img ** black_gamma

    return img.astype(np.float32)


# -----------------------------
# Cellpose inference
# -----------------------------
def predict_masks(img01,
                  use_gpu=True,
                  diameter=25,
                  cellprob_threshold=1.8,
                  flow_threshold=1.0):

    model = models.CellposeModel(gpu=use_gpu)

    masks, flows, styles = model.eval(
        img01,
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
    )

    return masks


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    image_path = "/media/NAS_R01_P1S1/RAW_DATA/Scripta/example_images/dcfa8e4f-a731-4dd1-89b7-ac285232aca8/images/r01c12/ch2/r01c12f01p01-ch02t01.tiff"

    print("Loading image...")
    img = io.imread(image_path)

    if img.ndim == 3:
        img = img[..., 0]   # use first channel if multi-channel

    print("Preprocessing...")
    img01 = preprocess(img)

    print("Running Cellpose...")
    masks = predict_masks(img01)

    print("Plotting result...")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Preprocessed")
    plt.imshow(img01, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Mask")
    plt.imshow(masks, cmap="tab20")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("test_output.png", dpi=150)