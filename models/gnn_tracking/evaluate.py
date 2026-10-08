"""
evaluate.py
===========
Track wells with the shipped 3-seed ensemble and score them against the annotation.

    python evaluate.py                      # the 3 test wells (the numbers in the README)
    python evaluate.py r08c13 r11c12        # any wells -- but the shipped weights were trained on the 12 development
                                            # wells, so only the test wells give a fair score for them
    python evaluate.py --weights my_weights # weights from another folder (see train.py --out)

The first run reads the raw frames of each well and caches the detections (about a minute per well).
"""
import os, sys
import numpy as np

from utils.dataset import HERE, TEST_WELLS, load_well
from models.gnn_tracking.graph import Video
from models.gnn_tracking.track import track_video, load_ensemble
from utils.metrics import gt_lookup, score

if __name__ == "__main__":
    args = sys.argv[1:]
    folder = "weights"
    if "--weights" in args:
        folder = args.pop(args.index("--weights") + 1); args.remove("--weights")
    models = load_ensemble(os.path.join(HERE, folder))
    res = {}
    for w in args or TEST_WELLS:
        video = Video(*load_well(w))
        gt = gt_lookup(video)
        res[w] = score(track_video(models, video), gt)
        adj = score(track_video(models, video, gap_round=False), gt)
        print(f"{w}: perfect {res[w]['perfect']:.1f}%  id switches {res[w]['id_switches']}  purity {res[w]['purity']:.3f}"
              f"   (round 1 only: {adj['perfect']:.1f}% / {adj['id_switches']})", flush=True)
    print(f"MEAN: perfect {np.mean([r['perfect'] for r in res.values()]):.1f}%  "
          f"id switches {np.mean([r['id_switches'] for r in res.values()]):.1f}  "
          f"purity {np.mean([r['purity'] for r in res.values()]):.3f}")
