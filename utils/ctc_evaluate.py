"""Association evaluation against the lab's identity-labelled CTC detections.

This is not official TRA/SEG: input detections and GT share the same masks.
Labels >=50000 have unknown identity and are excluded from identity scoring.
"""
import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tifffile
from skimage.measure import regionprops

from models import MODEL_MODULES
from utils.config import read_config, resolve_dataset
from utils.run_log import log_path, log_root, run_logged
from utils.ctc import sequence, write_csv
from utils.metrics import score

UNKNOWN_OFFSET = 50000


def read_track_table(path):
    tracks = {}
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 4:
            raise ValueError(f'{path}:{line_number}: expected id start end parent')
        tid, start, end, parent = map(int, values)
        if tid <= 0 or start < 0 or end < start or parent < 0 or tid in tracks:
            raise ValueError(f'{path}:{line_number}: invalid or duplicate track')
        if tid < UNKNOWN_OFFSET and parent != 0:
            raise ValueError('Current association evaluator supports no-division lab data only')
        tracks[tid] = dict(start=start, end=end, parent=parent)
    return tracks


def read_predictions(folder, frame_paths):
    with (folder / 'instance_tracks.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'frame_index','instance_id','track_id'} <= set(reader.fieldnames or []):
            raise ValueError('CSV needs frame_index, instance_id, track_id; window-only outputs cannot be evaluated')
        rows = list(reader)
    assign, frame_tracks = {}, set()
    for row in rows:
        t, label, pid = [int(row[k]) for k in ('frame_index','instance_id','track_id')]
        key = (t, label)
        if t not in frame_paths or label <= 0 or pid <= 0 or key in assign or (t,pid) in frame_tracks:
            raise ValueError(f'invalid/duplicate prediction at {key}')
        if row.get('mask_path'):
            mask = Path(row['mask_path'])
            if not mask.is_absolute():
                mask = folder / mask
            if mask.resolve() != frame_paths[t].resolve():
                raise ValueError('Association-only evaluation requires predictions from the same CTC masks')
        assign[key] = pid
        frame_tracks.add((t,pid))
    return assign


def evaluate_well(data_root, well, prediction_dir, output_dir, visualize=True):
    """Use exact (frame, mask instance) identity, without centroid matching or XML."""
    data_root, prediction_dir, output_dir = map(Path, (data_root,prediction_dir,output_dir))
    if output_dir.exists():
        raise ValueError(f'Evaluation output already exists: {output_dir}')
    if data_root.resolve() == output_dir.resolve() or data_root.resolve() in output_dir.resolve().parents:
        raise ValueError('Evaluation output must be outside the source CTC dataset')
    frames = sequence(data_root, well)
    table = read_track_table(data_root / f'{well}_GT' / 'TRA/man_track.txt')
    frame_paths = {t: mask for t,raw,mask in frames}
    assign = read_predictions(prediction_dir, frame_paths)
    gt, objects, all_keys, by_gt = {}, {}, set(), defaultdict(list)
    unknown_count = 0
    for t, raw, mask_path in frames:
        mask = tifffile.imread(mask_path)
        if mask.ndim != 2 or not np.issubdtype(mask.dtype,np.integer) or np.any(mask < 0):
            raise ValueError(f'invalid CTC mask: {mask_path}')
        with tifffile.TiffFile(raw) as image:
            if tuple(image.series[0].shape) != mask.shape:
                raise ValueError(f'image/mask shape mismatch: {raw}')
        local = []
        for p in regionprops(mask):
            tid = int(p.label)
            key = (t,tid)
            all_keys.add(key)
            local.append(dict(frame_index=t, instance_id=tid, x=float(p.centroid[1]), y=float(p.centroid[0])))
            if tid >= UNKNOWN_OFFSET:
                unknown_count += 1
                continue
            if tid not in table or not table[tid]['start'] <= t <= table[tid]['end']:
                raise ValueError(f'GT mask identity {tid} at frame {t} is absent/outside man_track.txt')
            gt[key] = tid
            by_gt[tid].append(key)
        objects[t] = local
    if set(assign) - all_keys:
        raise ValueError('Prediction instance IDs do not exist in the CTC masks')
    for tid, entry in table.items():
        if tid >= UNKNOWN_OFFSET:
            continue
        keys = by_gt.get(tid, [])
        if not keys or keys[0][0] != entry['start'] or keys[-1][0] != entry['end']:
            raise ValueError(f'GT track {tid}: mask observations disagree with table start/end')
    report = score(assign, gt)
    content = defaultdict(set)
    for key,pid in assign.items():
        if key in gt:
            content[pid].add(gt[key])
    mixed = {pid for pid,tids in content.items() if len(tids)>1}
    per_track, transitions, qualities, gaps = [], [], {}, defaultdict(Counter)
    for tid,keys in sorted(by_gt.items()):
        pids = [assign.get(key) for key in keys]
        observed = [pid for pid in pids if pid is not None]
        fragments = len(set(observed))
        missing = sum(pid is None for pid in pids)
        mixing = any(pid in mixed for pid in observed)
        recovered = not missing and fragments == 1 and not mixing
        qualities[tid] = 'recovered' if recovered else ('mixed' if mixing else ('missing' if missing else 'fragmented'))
        per_track.append(dict(gt_track_id=tid, start=table[tid]['start'], end=table[tid]['end'],
            gt_observations=len(keys), predicted_observations=len(observed), predicted_fragments=fragments,
            missing_predictions=missing, mixing=mixing, recovered=recovered,
            id_switches=sum(a!=b for a,b in zip(observed[:-1],observed[1:]))))
        for before,after in zip(keys[:-1],keys[1:]):
            previous,current = assign.get(before),assign.get(after)
            dt = after[0]-before[0]
            status = 'missing' if previous is None or current is None else ('retained' if previous==current else 'switched')
            transitions.append(dict(gt_track_id=tid, previous_frame=before[0], current_frame=after[0],
                dt=dt, previous_pred_track_id=previous, current_pred_track_id=current, status=status))
            gaps[dt]['total'] += 1
            gaps[dt]['retained'] += status == 'retained'
    matches, per_frame = [], []
    for t,raw,mask in frames:
        local = objects[t]
        for obj in local:
            label = obj['instance_id']
            known = label < UNKNOWN_OFFSET
            matches.append(dict(**obj, gt_track_id=label if known else '',
                pred_track_id=assign.get((t,label),''),
                status=qualities[label] if known else 'unknown_identity'))
        known = [obj for obj in local if obj['instance_id'] < UNKNOWN_OFFSET]
        per_frame.append(dict(frame_index=t, image_name=raw.name, gt_objects=len(known),
            predicted_gt_objects=sum((t,obj['instance_id']) in assign for obj in known),
            unknown_objects=len(local)-len(known)))
    total_links = len(transitions)
    report.update(status=('no_labelled_detections' if not gt else
                         ('complete' if len(set(gt)&set(assign))==len(gt) else 'partial')),
        metric_scope='association_only_on_existing_lab_ctc_detections', official_ctc_metrics=False,
        sequence_id=well, data_root=str(data_root.resolve()), prediction_dir=str(prediction_dir.resolve()),
        n_frames=len(frames), gt_observations=len(gt), predicted_gt_observations=len(set(gt)&set(assign)),
        observation_coverage=len(set(gt)&set(assign))/len(gt) if gt else None,
        unknown_observations=unknown_count, unscored_unknown_predictions=sum(key not in gt for key in assign),
        n_pred_tracks_total=len(set(assign.values())), mixed_pred_tracks=len(mixed),
        identity_link_recall=sum(r['status']=='retained' for r in transitions)/total_links if total_links else None,
        identity_link_recall_by_dt={str(dt):dict(total=c['total'], retained=c['retained'],
            recall=c['retained']/c['total']) for dt,c in sorted(gaps.items())},
        gt_track_table=str((data_root/f'{well}_GT/TRA/man_track.txt').resolve()),
        gt_gaps='Only observed mask detections are scored; table intervals do not invent missing detections')
    output_dir.mkdir(parents=True)
    (output_dir/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    write_csv(output_dir/'per_gt_track.csv',per_track,
        ['gt_track_id','start','end','gt_observations','predicted_observations','predicted_fragments',
         'missing_predictions','mixing','recovered','id_switches'])
    write_csv(output_dir/'transitions.csv',transitions,
        ['gt_track_id','previous_frame','current_frame','dt','previous_pred_track_id','current_pred_track_id','status'])
    write_csv(output_dir/'instance_matches.csv',matches,
        ['frame_index','instance_id','x','y','gt_track_id','pred_track_id','status'])
    write_csv(output_dir/'per_frame.csv',per_frame,
        ['frame_index','image_name','gt_objects','predicted_gt_objects','unknown_objects'])
    if visualize:
        from utils.visualization import render_ctc_comparison
        destination = output_dir/'visualizations'
        destination.mkdir()
        rows_by_frame = defaultdict(list)
        for row in matches:
            rows_by_frame[row['frame_index']].append(row)
        for t,raw,mask in frames:
            render_ctc_comparison(raw,mask,rows_by_frame[t],destination/f'{raw.stem}_comparison.png')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,help='Default: project config.json')
    parser.add_argument('--dataset',help='Dataset name from the config, or a folder under data_dir (default: config dataset)')
    parser.add_argument('--data-root',type=Path,help='Explicit CTC root; overrides the dataset root')
    parser.add_argument('--tracking-output',type=Path,help='Saved test output of one run (required unless --compare)')
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--wells',nargs='+')
    parser.add_argument('--visualize',choices=['all','none'],default='all',help='Per-frame comparison PNGs')
    parser.add_argument('--plots',action=argparse.BooleanOptionalAction,default=True,
                        help='Metric summary figure (PDF/SVG/PNG + source data) under <output-root>/figures')
    parser.add_argument('--compare',type=Path,nargs='+',metavar='EVALUATION',
                        help='Compare existing evaluate.py outputs of different runs on the same sequences')
    parser.add_argument('--labels',nargs='+',help='Display names for --compare runs (default: model and version)')
    parser.add_argument('--overwrite',action='store_true',
                        help='Replace an existing output-root that holds an earlier evaluate.py result')
    args=parser.parse_args(argv)
    cfg,base=read_config(args.config)
    if args.compare:
        if args.labels and len(args.labels)!=len(args.compare):
            parser.error('--labels needs one name per --compare folder')
        clear_output(parser,args.output_root,args.overwrite,args.compare)
        log=log_path(log_root(args.output_root,base/cfg['output_root']),'evaluate',f'{args.output_root.resolve().name}_compare')
        return run_logged(log,compare_runs,args,argv=argv)
    if not args.tracking_output:
        parser.error('--tracking-output is required unless --compare is given')
    try:
        dataset=resolve_dataset(cfg,base,args.dataset,args.data_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    args.data_root,args.dataset_name=dataset['root'],dataset['name']
    wells=args.wells or sorted(p.parent.name for p in args.tracking_output.glob('*/instance_tracks.csv'))
    if not wells or len(wells)!=len(set(wells)):
        parser.error('no wells found or duplicate wells supplied')
    clear_output(parser,args.output_root,args.overwrite,[args.data_root,args.tracking_output])
    return run_logged(evaluation_log(args.tracking_output,args.output_root,base/cfg['output_root'],dataset['name']),
                      evaluate_wells,args,wells,argv=argv)


def clear_output(parser,output_root,overwrite,protected):
    """Refuse an existing output-root unless --overwrite; then delete it only if it is an earlier evaluate.py
    result (metrics.json / comparison.json / run.json at its root or per-sequence metrics.json) and contains none
    of the protected inputs."""
    if not output_root.exists():
        return
    if not overwrite:
        parser.error('output-root already exists; pass --overwrite to replace it or select a new directory')
    root=output_root.resolve()
    markers=('metrics.json','comparison.json','run.json')
    if not root.is_dir() or not (any((root/m).is_file() for m in markers) or any(root.glob('*/metrics.json'))):
        parser.error(f'--overwrite only replaces an earlier evaluate.py output; {root} does not look like one')
    for path in protected:
        path=Path(path).resolve()
        if path==root or root in path.parents:
            parser.error(f'--overwrite refused: {root} contains input {path}')
    shutil.rmtree(root)


def evaluation_log(tracking_output,output_root,default_output_root,dataset):
    """Runs laid out as <root>/<dataset>/<model>/<version>/test log next to their test log; others use the
    evaluation folder name under the output root (see log_root)."""
    parts=tracking_output.resolve().parts
    if len(parts)>=5 and parts[-1]=='test' and parts[-3] in MODEL_MODULES:
        return log_path(Path(*parts[:-4]),parts[-3],f'{parts[-2]}_evaluate_{parts[-4]}')
    return log_path(log_root(output_root,default_output_root),'evaluate',f'{output_root.resolve().name}_{dataset}')


def run_info(tracking_output,dataset):
    """Model/version of a run laid out as <root>/<dataset>/<model>/<version>/test (used for figure labels)."""
    parts=tracking_output.resolve().parts
    info=dict(tracking_output=str(tracking_output.resolve()),dataset=dataset)
    if len(parts)>=5 and parts[-1]=='test' and parts[-3] in MODEL_MODULES:
        info.update(model=parts[-3],version=parts[-2],dataset=parts[-4])
    return info


def compare_runs(args):
    from utils.metric_plots import load_evaluation, plot_metrics
    runs=[load_evaluation(folder,label) for folder,label in zip(args.compare,args.labels or [None]*len(args.compare))]
    wells=[sorted(run['wells']) for run in runs]
    if any(w!=wells[0] for w in wells):
        raise ValueError('Compared evaluations must cover the same sequences: '+
                         '; '.join(f"{run['label']}: {' '.join(w)}" for run,w in zip(runs,wells)))
    args.output_root.mkdir(parents=True)
    (args.output_root/'comparison.json').write_text(json.dumps(dict(
        evaluations=[str(Path(f).resolve()) for f in args.compare],labels=[run['label'] for run in runs],
        sequences=wells[0]),indent=2),encoding='utf-8')
    for path in plot_metrics(runs,args.output_root/'figures'/'metrics_comparison'):
        print('saved',path,flush=True)
    return 0


def evaluate_wells(args,wells):
    reports={}
    for well in wells:
        reports[well]=evaluate_well(args.data_root,well,args.tracking_output/well,
            args.output_root/well,args.visualize=='all')
        print(well,{key:reports[well][key] for key in
            ('perfect','id_switches','purity','observation_coverage','identity_link_recall')},flush=True)
    (args.output_root/'metrics.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    info=run_info(args.tracking_output,args.dataset_name)
    (args.output_root/'run.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    if args.plots:
        from utils.metric_plots import load_evaluation, plot_metrics
        run=load_evaluation(args.output_root)
        for path in plot_metrics([run],args.output_root/'figures'/'metrics_summary',title=f"{run['label']} · {info['dataset']}"):
            print('saved',path,flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
