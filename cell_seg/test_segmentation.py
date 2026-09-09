r"""Test Cellpose cpsam_v2 using the tracking pipeline's preprocessing.

Example (PowerShell, from the project root):
    & D:\MiniConda\envs\cellpose\python.exe cell_seg/test_segmentation.py
Outputs are written beside this script. No tracking is performed.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Windows Conda MKL and PyTorch can load separate Intel OpenMP runtimes.
# Keep MKL sequential to avoid the duplicate runtime; CUDA remains available.
# This must precede imports of NumPy, PyTorch, and their dependencies.
if sys.platform == 'win32':
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Single TIFF file; prompts when omitted')
    parser.add_argument('--model', default='cpsam_v2',
                        help='Built-in cpsam_v2 (default) or a local model file path')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--diameter', type=float, default=25)
    parser.add_argument('--cellprob-threshold', type=float, default=1.8)
    parser.add_argument('--flow-threshold', type=float, default=1.0)
    args = parser.parse_args()
    if args.input is None:
        args.input = Path(input('Enter single TIFF file path: ').strip().strip('\"').strip("'"))
    model_source = args.model
    if model_source != 'cpsam_v2':
        model_path = Path(model_source).expanduser()
        if not model_path.is_file():
            parser.error(f'Model file does not exist: {model_path}')
        model_source = str(model_path.resolve())
    if not args.input.is_file() or args.input.suffix.lower() not in {'.tif', '.tiff'}:
        parser.error(f'Please provide an existing .tif or .tiff file: {args.input}')

    import numpy as np
    import torch
    import tifffile
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from skimage import io
    from skimage.color import label2rgb
    from skimage.segmentation import find_boundaries
    from cellpose import models
    from main_tracking.segmentation import preprocess
    from main_tracking.mask_area_filter import filter_small_instances_by_mean, compute_mask_area_stats

    output = Path(__file__).resolve().parent
    files = [args.input]

    gpu = not args.cpu and torch.cuda.is_available()
    print(f'Loading model: {model_source} | GPU={gpu}', flush=True)
    model = models.CellposeModel(gpu=gpu, pretrained_model=model_source)
    report = {
        'model': model_source, 'gpu': gpu,
        'parameters': {'diameter': args.diameter, 'cellprob_threshold': args.cellprob_threshold,
                       'flow_threshold': args.flow_threshold, 'p_low': 1, 'p_high': 99,
                       'bg_sigma': 80, 'black_gamma': 1.8, 'auto_min_area_fraction': 0.25},
        'images': [],
    }
    for index, path in enumerate(files, 1):
        start = time.perf_counter()
        raw = io.imread(str(path))
        # Match existing first-channel behavior for channel-last color images.
        # Reject stacks rather than silently treating a time/Z axis as color.
        if raw.ndim == 3 and raw.shape[-1] in (3, 4):
            raw = raw[..., 0]
        raw = np.squeeze(raw)
        if raw.ndim != 2:
            raise ValueError(f'{path}: expected 2D grayscale or RGB image, got {raw.shape}')
        if not np.isfinite(raw).all():
            raise ValueError(f'{path}: image contains NaN or infinity')
        processed = preprocess(raw, p_low=1, p_high=99, bg_sigma=80, black_gamma=1.8)
        masks, _, _ = model.eval(processed, diameter=args.diameter,
                                 cellprob_threshold=args.cellprob_threshold,
                                 flow_threshold=args.flow_threshold)
        masks, min_area, before = filter_small_instances_by_mean(masks, fraction_of_mean=0.25)
        count = compute_mask_area_stats(masks).count
        prefix = output / f'{index:03d}_{path.stem}'
        dtype = np.uint16 if masks.max() <= 65535 else np.uint32
        tifffile.imwrite(str(prefix) + '_mask.tiff', masks.astype(dtype))

        lo, hi = np.percentile(raw, (1, 99))
        display = np.clip((raw.astype(np.float32) - lo) / max(float(hi - lo), 1e-8), 0, 1)
        overlay = label2rgb(masks, image=display, alpha=0.3, bg_label=0, saturation=0)
        overlay[find_boundaries(masks, mode='inner')] = (1, 1, 0)
        io.imsave(str(prefix) + '_overlay.png', (overlay * 255).astype(np.uint8), check_contrast=False)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for ax, data, title in zip(axes, [display, processed, overlay],
                                  ['Original (1-99% display)', 'Tracking preprocessing', f'Overlay: {count} cells']):
            ax.imshow(data, cmap='gray', vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis('off')
        fig.savefig(str(prefix) + '_comparison.png', dpi=160)
        plt.close(fig)
        report['images'].append({'input': str(path.resolve()), 'shape': list(raw.shape),
                                 'instances_before_filter': before.count, 'instances': count,
                                 'min_area': min_area, 'seconds': round(time.perf_counter() - start, 2),
                                 'output_prefix': str(prefix)})
        (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'[{index}/{len(files)}] {path.name}: {count} cells; min_area={min_area}', flush=True)
    print(f'Done. Outputs: {output}', flush=True)


if __name__ == '__main__':
    main()
