from pathlib import Path

from PIL import Image, ImageSequence
import numpy as np


def convert_tiff_to_png(src_dir: Path, dst_dir: Path) -> None:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = list(src_dir.rglob("*.tif")) + list(src_dir.rglob("*.tiff"))
    for tif_file in tiff_files:
        relative = tif_file.relative_to(src_dir)
        out_file = (dst_dir / relative).with_suffix(".png")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Open TIFF and normalize pixel values to 0-255 before saving as PNG.
        with Image.open(tif_file) as img:
            frame = next(ImageSequence.Iterator(img))
            arr = np.array(frame).astype(np.float32)
            minv = arr.min()
            maxv = arr.max()
            if maxv > minv:
                norm = (arr - minv) / (maxv - minv) * 255.0
            else:
                norm = np.zeros_like(arr)

            norm_u8 = norm.astype(np.uint8)
            if norm_u8.ndim == 2:
                out_img = Image.fromarray(norm_u8, mode="L")
            else:
                out_img = Image.fromarray(norm_u8)
            out_img.save(out_file, format="PNG")

    print(f"Converted {len(tiff_files)} TIFF files to PNG in: {dst_dir}")


if __name__ == "__main__":
    data_path = Path(
        "/media/NAS_R01_P1S1/RAW_DATA/Scripta/example_images/"
        "dcfa8e4f-a731-4dd1-89b7-ac285232aca8/images/r01c12/ch2/masks_out"
    )
    output_path = Path(
        "/media/NAS_R01_P1S1/RAW_DATA/Scripta/example_images/"
        "dcfa8e4f-a731-4dd1-89b7-ac285232aca8/images/r01c12/ch2/mask_out_png"
    )
    convert_tiff_to_png(data_path, output_path)
