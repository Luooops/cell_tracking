"""Input/output adapters. Association, network and solver implementations stay in models/."""
import json
import os
from pathlib import Path
from models import get_model
from utils.config import save_config


def run_test(a):
    if a.model == 'motion_tracking':
        return motion_test(a)
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
            df['track_id'] = [assign[(fi, i)] for fi, t in enumerate(video.frames) for i in range(len(video.rows[t]))]
            destination.mkdir()
            if parent is not None:
                (destination / 'lineage.json').write_text(json.dumps(parent), encoding='utf-8')
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


def motion_test(a):
    if not a.windows_dir or not a.weights:
        raise ValueError('Motion requires --windows-dir and --weights; each existing NPZ window is decoded independently, without new video fusion.')
    import numpy as np
    import torch
    from utils.ctc import write_csv
    module = get_model(a.model)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    checkpoint = torch.load(a.weights, map_location=device, weights_only=False)
    model = module.MotionAssociation(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    paths = sorted(a.windows_dir.rglob('*.npz'))
    if not paths:
        raise ValueError('No NPZ windows found')
    save_config(a)
    with torch.no_grad():
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                coords = torch.as_tensor(data['coords'], dtype=torch.float32, device=device)
                features = torch.as_tensor(data['features'], dtype=torch.float32, device=device)
            if len(coords) > a.params['max_tokens']:
                raise ValueError(f'Window exceeds max_tokens: {path}; no automatic truncation or fusion')
            features = (features - checkpoint['feature_mean'].to(device)) / checkpoint['feature_std'].to(device)
            edges, ids = module.decode_tracks(model(coords, features), coords, threshold=a.params['threshold'])
            dest = a.destination / path.relative_to(a.windows_dir).with_suffix('')
            dest.mkdir(parents=True)
            points = coords.cpu().numpy()
            rows = [dict(instance_index=i, frame_index=int(c[0]), x=float(c[1]), y=float(c[2]), track_id=ids[i]) for i, c in enumerate(points)]
            write_csv(dest / 'instance_tracks.csv', rows, ['instance_index','frame_index','x','y','track_id'])
            (dest / 'edges.json').write_text(json.dumps(edges), encoding='utf-8')
    return 0
