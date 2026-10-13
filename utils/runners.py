"""Input/output adapters. Association, network and solver implementations stay in models/."""
import json
import os
from pathlib import Path
from models import get_model
from utils.config import save_config


def run_test(a):
    if a.model == 'motion_tracking':
        from models.motion_tracking.inference import run
        return run(a)
    os.environ['LAB_TRACKER_DATA'] = str(a.data_root)
    os.environ['LAB_TRACKER_CACHE'] = str(a.destination / 'cache')
    if a.device != 'auto':
        os.environ['LAB_TRACKER_DEVICE'] = a.device
    from utils.ctc import sequence, detections, manifest, write_csv
    from utils.metrics import score
    from utils import dataset
    dataset.LAB_ROOT = str(a.data_root)
    dataset.CACHE = str(a.destination / 'cache')
    if not a.wells:
        raise ValueError('Select at least one well')
    for well in a.wells:
        sequence(a.data_root, well)
    module = get_model(a.model)
    if a.model in {'gnn_tracking', 'gnn_division'}:
        from models.gnn_tracking.track import load_ensemble
        model = load_ensemble(str(a.weights))
    elif a.model == 'trackastra':
        from trackastra.model import Trackastra
        import torch
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
        model = Trackastra.from_pretrained(a.params['pretrained'], device=device)
        model._pretrained_name = a.params['pretrained']
    else:
        model = None
    save_config(a)
    reports = {}
    for well in a.wells:
        destination = a.destination / well
        if a.model in {'gnn_tracking', 'gnn_division'}:
            from utils.dataset import load_well
            from models.gnn_tracking.graph import Video
            from utils.metrics import gt_lookup
            video = Video(*load_well(well))
            parent = None
            if a.model == 'gnn_division':
                assign, parent = module.track_video_division(model, video, **a.params)
            else:
                assign = module.track_video(model, video, gap_round=a.params['gap_round'])
            reports[well] = score(assign, gt_lookup(video))
            df = video.df.copy()
            df = df.rename(columns={'track_id': 'instance_id', 'frame': 'frame_index'})
            # GNN track ids start at 0; exported ids are shifted by 1 because 0 is background in CTC (evaluate.py).
            df['track_id'] = [assign[(fi, i)] + 1 for fi, t in enumerate(video.frames) for i in range(len(video.rows[t]))]
            destination.mkdir()
            if parent is not None:
                lineage = {int(d) + 1: int(m) + 1 for d, m in parent.items()}
                (destination / 'lineage.json').write_text(json.dumps(lineage), encoding='utf-8')
            df.to_csv(destination / 'instance_tracks.csv', index=False)
        else:
            frames, rows, gt = detections(a.data_root, well)
            if a.model == 'trackastra':
                # The old tracker still consumes exactly its original manifest schema.
                source = a.destination / 'inputs' / well
                manifest(source, a.data_root, well, frames, rows)
                module.track_sequence(source, a.data_root / well, a.destination, model,
                                          a.params['mode'], False, False)
                rows = module.csv_read(destination / 'instance_tracks.csv')
            else:
                import tifffile
                from models.classical_tracking.tracking_v0 import extract_detections_from_mask
                params = dict(a.params)
                gap = params.pop('gap_close_max_gap', 0)
                distance = params.pop('gap_close_max_distance', 30)
                cost = params.pop('max_close_cost', 12)
                cls = module.MitosisTracker if a.model == 'mitosis' else module.SimpleCellTrackerV3
                tracker = cls(**params)
                for frame, _, mask in sequence(a.data_root, well):
                    tracker.update(extract_detections_from_mask(tifffile.imread(mask), frame))
                tracker.finish_all()
                tracks = tracker.get_all_tracks()
                if gap > 0:
                    tracks = module.gap_close_tracks(tracks, max_gap=gap, max_dist=distance,
                        max_area_ratio=params.get('max_area_ratio', 1.8), max_shape_ratio=params.get('max_shape_ratio', 1.8),
                        max_angle_diff_deg=120, max_close_cost=cost)
                destination.mkdir()
                from models.classical_tracking.tracking_v0 import export_tracks_to_csv
                df = export_tracks_to_csv(tracks, str(destination / 'tracks.csv'))
                df['instance_id'], df['frame_index'] = df['label'], df['frame']
                df.to_csv(destination / 'instance_tracks.csv', index=False)
                rows = df.to_dict('records')
                write_csv(destination / 'frames.csv', frames, list(frames[0]))
                if a.model == 'mitosis':
                    (destination / 'lineage.json').write_text(json.dumps(tracker.parent_of), encoding='utf-8')
            assign = {(int(r['frame_index']), int(r['instance_id'])): int(r['track_id']) for r in rows}
            reports[well] = score(assign, gt) if gt else {'status': 'no_labelled_detections'}
        (destination / 'metrics.json').write_text(json.dumps(reports[well], indent=2), encoding='utf-8')
        print(well, reports[well])
    (a.destination / 'metrics.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    return 0
