from pathlib import Path

import numpy as np
from PIL import Image


def _load_frame(frame: Image.Image) -> Image.Image:
    return frame.convert("RGBA")


def convert_png_series_to_gif(
    src_dir: Path,
    output_gif: Path,
    duration_ms: int = 100,
    loop: int = 0,
    recursive: bool = True,
) -> None:
    src_dir = Path(src_dir)
    output_gif = Path(output_gif)
    output_gif.parent.mkdir(parents=True, exist_ok=True)

    pattern_iter = src_dir.rglob if recursive else src_dir.glob
    png_files = sorted(pattern_iter("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {src_dir}")

    loaded_frames: list[Image.Image] = []
    canvas_width = 0
    canvas_height = 0

    for png_file in png_files:
        with Image.open(png_file) as img:
            loaded_frame = _load_frame(img)
            loaded_frames.append(loaded_frame)
            canvas_width = max(canvas_width, loaded_frame.width)
            canvas_height = max(canvas_height, loaded_frame.height)

    frames: list[Image.Image] = []
    canvas_size = (canvas_width, canvas_height)
    for frame in loaded_frames:
        if frame.size != canvas_size:
            padded = Image.new("RGBA", canvas_size, color=(0, 0, 0, 0))
            padded.paste(frame, (0, 0), frame)
            frame = padded
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE))

    first_frame, *other_frames = frames
    first_frame.save(
        output_gif,
        format="GIF",
        save_all=True,
        append_images=other_frames,
        duration=duration_ms,
        loop=loop,
        optimize=False,
    )

    print(f"Saved {len(frames)} frames to GIF: {output_gif}")


def main() -> None:
    src_dir = Path(
        "./z_slices_png"
    )
    output_gif = Path("./output_spores_png.gif")
    convert_png_series_to_gif(
        src_dir,
        output_gif,
        duration_ms=100,
        loop=0,
        recursive=True,
    )


if __name__ == "__main__":
    main()