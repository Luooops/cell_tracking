"""Full-video association with overlap fusion and a sparse global path cover."""
import json

from models.motion_tracking import model as module
from utils.config import save_config
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.special import logsumexp
from utils.ctc import sequence, write_csv
from models.motion_tracking.prepare_npz import extract_frames, FEATURE_NAMES


def video_windows(coords, times, window_frames, max_gap, max_tokens):
    """Each target has its complete candidate past; never silently drop gap edges."""
    if window_frames < max_gap + 1:
        raise ValueError('window_frames must be at least checkpoint max_gap + 1')
    seen = set()
    for t in times:
        required_start = max(times[0], t - max_gap)
        start = max(times[0], min(required_start, times[-1] - window_frames + 1))
        end = min(times[-1], start + window_frames - 1)
        while True:
            selected = np.flatnonzero((coords[:, 0] >= start) & (coords[:, 0] <= end))
            if len(selected) <= max_tokens:
                break
            if end > t:
                end -= 1
            elif start < required_start:
                start += 1
            else:
                raise ValueError(f'Full candidate horizon {start}..{end} exceeds max_tokens={max_tokens}; '
                                 'increase max_tokens. Shortening it would lose gap candidates.')
        if (start, end) not in seen:
            seen.add((start, end))
            yield start, end, selected


def decode_global(coords, candidates, threshold):
    """Exact one-predecessor/one-successor path cover using sparse assignment."""
    if not 0 < threshold < 1:
        raise ValueError('threshold must lie strictly between 0 and 1')
    n = len(coords)
    accepted = [c for c in candidates if c['probability'] >= threshold and c['log_odds'] > 0]
    edges = []
    if n:
        # Positive shift keeps dummy edges present in SciPy's sparse graph.
        base = max([c['log_odds'] for c in accepted], default=0) + 1
        rows = [c['source'] for c in accepted] + list(range(n))
        cols = [c['target'] for c in accepted] + list(range(n, 2*n))
        costs = [base - c['log_odds'] for c in accepted] + [base] * n
        graph = coo_matrix((costs, (rows, cols)), shape=(n, 2*n)).tocsr()
        sources, targets = min_weight_full_bipartite_matching(graph)
        edges = [(int(i), int(j)) for i, j in zip(sources, targets) if j < n]
    parent = {j: i for i, j in edges}
    ids, next_id = [0] * n, 1
    for j in sorted(range(n), key=lambda i: (coords[i, 0], i)):
        if j in parent:
            ids[j] = ids[parent[j]]
        else:
            ids[j], next_id = next_id, next_id + 1
    return edges, ids


def track_video(model, coords, features, times, mean, std, window_frames=6, max_tokens=2048, threshold=0.5):
    """Fuse edge-vs-null logits only when all predecessor candidates are present."""
    if not times or times != list(range(times[0], times[-1] + 1)):
        raise ValueError('video frames must be consecutive, including empty detection frames')
    if len(coords) != len(features) or not np.isfinite(coords).all() or not np.isfinite(features).all():
        raise ValueError('invalid video coordinates/features')
    if len(coords) and not np.isin(coords[:, 0], times).all():
        raise ValueError('detections reference frames outside the video')
    max_gap = model.config['max_gap']
    device = next(model.parameters()).device
    mean, std = torch.as_tensor(mean, device=device), torch.as_tensor(std, device=device)
    if (mean.shape != (features.shape[1],) or std.shape != mean.shape or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('invalid feature scaler')
    sums, window_count = {}, 0
    model.eval()
    with torch.no_grad():
        for start, end, selected in video_windows(coords, times, window_frames, max_gap, max_tokens):
            if not len(selected):
                continue
            c = torch.as_tensor(coords[selected], dtype=torch.float32, device=device)
            f = (torch.as_tensor(features[selected], dtype=torch.float32, device=device) - mean) / std
            out = model(c, f)
            eligible = coords[selected, 0] - max_gap >= start
            eligible |= start == times[0]
            valid = out['valid'].cpu().numpy() & eligible[None, :]
            source, target = np.nonzero(valid)
            odds = (out['logits'] - out['null_logits'][None]).cpu().numpy()[source, target]
            if not np.isfinite(odds).all():
                raise ValueError('nonfinite association scores')
            for i, j, value in zip(selected[source], selected[target], odds):
                key = (int(i), int(j))
                total, count = sums.get(key, (0.0, 0))
                sums[key] = (total + float(value), count + 1)
            window_count += 1
    incoming = {}
    for (i, j), (total, count) in sums.items():
        incoming.setdefault(j, []).append(dict(source=i, target=j, log_odds=total/count, windows=count))
    candidates = []
    for j, options in incoming.items():
        denominator = logsumexp([0.0] + [o['log_odds'] for o in options])
        for option in options:
            option['probability'] = float(np.exp(option['log_odds'] - denominator))
            candidates.append(option)
    candidates.sort(key=lambda c: (c['source'], c['target']))
    edges, ids = decode_global(coords, candidates, threshold)
    chosen = set(edges)
    for candidate in candidates:
        candidate['selected'] = (candidate['source'], candidate['target']) in chosen
    return edges, ids, candidates, window_count


def run(a):
    if not a.weights:
        raise ValueError('Motion requires --weights')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    checkpoint = torch.load(a.weights, map_location=device, weights_only=False)
    if 'feature_names' in checkpoint and checkpoint['feature_names'] != FEATURE_NAMES:
        raise ValueError('checkpoint feature order is incompatible with inference')
    model = module.MotionAssociation(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if a.windows_dir:
        save_config(a)
        return run_windows(a, model, checkpoint, device)
    if not a.wells:
        raise ValueError('Select at least one well')
    save_config(a)
    reports = {}
    for well in a.wells:
        relative = Path(well)
        triples = sequence(a.data_root, well)
        frames = [dict(sequence_id=well, frame_index=t, time_index=t, image_name=raw.name,
            source_relative_path=raw.name, mask_path=str(mask), status='complete') for t, raw, mask in triples]
        summary = dict(sequence_id=well, dataset=a.dataset, input_dir=str(a.data_root / relative))
        coords, features, labels, times = extract_frames(triples)
        import tifffile
        for frame, (_, raw, _) in zip(frames, triples):
            with tifffile.TiffFile(raw) as image:
                height, width = image.series[0].shape
            frame['height'], frame['width'] = height, width
        edges, ids, candidates, windows = track_video(model, coords, features, times,
            checkpoint['feature_mean'], checkpoint['feature_std'], a.params['window_frames'],
            a.params['max_tokens'], a.params['threshold'])
        dest = a.destination / relative
        dest.mkdir(parents=True, exist_ok=True)
        by_time = {int(f['frame_index']): f for f in frames}
        frame_order = {t: i for i, t in enumerate(times)}
        rows = [dict(instance_index=i, instance_id=int(label), frame_index=int(c[0]), frame=frame_order[int(c[0])],
            image_name=by_time[int(c[0])]['image_name'], mask_path=by_time[int(c[0])]['mask_path'],
            x=float(c[1]), y=float(c[2]), track_id=ids[i]) for i, (c, label) in enumerate(zip(coords, labels))]
        columns = ['instance_index','instance_id','frame_index','frame','image_name','mask_path','x','y','track_id']
        write_csv(dest / 'instance_tracks.csv', rows, columns)
        write_csv(dest / 'frames.csv', frames, list(frames[0]))
        (dest / 'edges.json').write_text(json.dumps(edges), encoding='utf-8')
        write_csv(dest / 'association_candidates.csv', candidates,
            ['source','target','log_odds','probability','windows','selected'])
        summary.update(status='complete', model='motion_tracking', parameters=a.params,
            n_frames=len(times), n_detections=len(coords), n_tracks=len(set(ids)), n_windows=windows,
            gap_edges=sum(coords[j,0]-coords[i,0]>1 for i,j in edges), weights=str(a.weights),
            fusion='mean edge-vs-null logits with complete candidate history',
            uncertainty='association probabilities are not calibrated error rates')
        # Convert numpy counts before JSON serialization.
        summary['gap_edges'] = int(summary['gap_edges'])
        (dest / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        from utils.ctc_evaluate import evaluate_well
        report = evaluate_well(a.data_root, well, dest, dest / 'evaluation', a.params['visualize'])
        reports[str(relative)] = report
        (dest / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if a.params['visualize']:
            from utils.visualization import render_tracking_frames
            render_tracking_frames(triples, rows, dest / 'visualizations')
        print(relative, {'frames':len(times), 'tracks':len(set(ids)), **report}, flush=True)
    (a.destination / 'metrics.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    return 0


def run_windows(a, model, checkpoint, device):
    paths = sorted(a.windows_dir.rglob('*.npz'))
    if not paths:
        raise ValueError('No NPZ windows found')
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
