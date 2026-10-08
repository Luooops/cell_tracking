> Historical model documentation. For current commands and paths, use the root README.md.

# Tracking pipeline

Run `python models/classical_tracking/main.py --input-dir E:/aws_gt_data_refine --batch --output-root models/classical_tracking/outputs_refined_model`.

The single entry point reruns Cellpose segmentation, extracts detections, performs adaptive association and exports masks, overlays, frames.csv, instance_tracks.csv and summary.json. The directory and CSV schemas are unchanged.

Defaults: local cp4_20260721_210258 weights, raw image with Cellpose normalization, three identical grayscale channels, cellprob=0, flow=0.4, no extra area filtering, model-native diameter, deformable association, motion-scale=0.5, max-distance=45, max-lost=3. Extra gap closing is disabled (gap-close-max-gap=0). Set a positive value to enable the legacy gap-closing postprocessor. Use --no-deformable for legacy association.

Completed nonempty sequence destinations are not overwritten. Use a fresh output root. No standalone retrack entry point remains.
