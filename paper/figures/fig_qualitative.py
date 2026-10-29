"""Figure 4 (full width): one test-well region over consecutive frames; rows GT, ours, Trackastra ft.

Contours are coloured by identity: GT by cell; a predicted track takes the colour of the GT cell of its first
known detection in the shown frames, so a correct track keeps the GT colour and an identity error shows as a change.
Markers on the incoming link of a detection: red cross = false link, red ring = missed link.
Ours also shows p of its uncertain chosen links (decision band, links.csv `uncertain`).

The region is chosen automatically (Trackastra errors where ours is right, with an uncertain link of ours inside);
candidates go to the legend file, and --well --frame --center override the choice:
    python fig_qualitative.py --well r07c07 --frame 12 --center 1040 880
"""
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib import patheffects
from skimage.segmentation import find_boundaries

import data
import style
from utils.ctc import sequence
from utils.lineage import UNKNOWN_OFFSET, read_lineage

ROWS = [('gt', 'Ground truth'), ('ours', 'Ours'), ('ta_ft', 'Trackastra ft')]
FRAMES = 4                  # shown frames t0 .. t0+3
CROP = (300, 450)           # rows, cols (pixels)
EDGE_WIDTH = 4             # contour width (pixels of the 2160 x 2160 image)
UNKNOWN = '#6E6E6E'         # detections without GT identity
PALETTE = ['#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', '#911EB4', '#46F0F0', '#F032E6', '#BCF60C',
           '#FABEBE', '#008080', '#E6BEFF', '#9A6324', '#FFFAC8', '#AAFFC3', '#808000', '#FFD8B1', '#00A0FF']


def load_well(well):
    out = {k: pd.read_csv(data.track_folder(k, well) / 'instance_tracks.csv') for k in ('ours', 'ta_ft')}
    ref = out['ours']
    observations = list(zip(ref.frame_index.astype(int), ref.instance_id.astype(int)))
    lineage = read_lineage(data.SCRIPTA / f'{well}_GT' / 'TRA' / 'man_track.txt', observations)
    gt = {(l.src, l.dst) for l in lineage.links if l.kind == 'continuation'}
    links = pd.read_csv(data.track_folder('ours', well) / 'links.csv')
    uncertain = {((r.src_frame, r.src_label), (r.dst_frame, r.dst_label)): r.p for r in links.itertuples() if r.uncertain}
    centre = {(int(r.frame_index), int(r.instance_id)): (r.y, r.x) for r in ref.itertuples()}
    return out, lineage, gt, uncertain, centre


def incoming_errors(tracks, gt):
    """{detection: 'false' | 'missed'} for the incoming link of each detection (false wins when both)."""
    pred = data.predicted_links(tracks)
    errors = {}
    for s, d in gt - pred:
        errors[d] = 'missed'
    for s, d in pred - gt:
        if s[1] < UNKNOWN_OFFSET and d[1] < UNKNOWN_OFFSET:
            errors[d] = 'false'
    return errors


def candidates(top=10):
    """Regions where Trackastra ft errs and ours is right, scored over the incoming links of frames t0+1 .. t0+3."""
    found = []
    for well in data.TEST_WELLS:
        out, _, gt, uncertain, centre = load_well(well)
        err = {k: incoming_errors(out[k], gt) for k in out}
        unc = defaultdict(list)
        for (_, d), p in uncertain.items():
            unc[d[0]].append(centre[d])
        n_frames = max(f for f, _ in centre) + 1
        size = max(y for y, _ in centre.values()), max(x for _, x in centre.values())
        for d, kind in err['ta_ft'].items():
            if d in err['ours']:
                continue
            if not (CROP[0] / 2 <= centre[d][0] <= size[0] - CROP[0] / 2 and
                    CROP[1] / 2 <= centre[d][1] <= size[1] - CROP[1] / 2):
                continue                          # keep the crop inside the image, centred on the error
            t0 = min(max(d[0] - 1, 0), n_frames - FRAMES)
            cy, cx = centre[d]
            inside = lambda e: (t0 < e[0] < t0 + FRAMES and abs(centre[e][0] - cy) < CROP[0] / 2
                                and abs(centre[e][1] - cx) < CROP[1] / 2)
            ta = sum(map(inside, err['ta_ft']))
            ours = sum(map(inside, err['ours']))
            n_unc = sum(abs(y - cy) < CROP[0] / 2 and abs(x - cx) < CROP[1] / 2
                        for f in range(t0 + 1, t0 + FRAMES) for y, x in unc[f])
            found.append(dict(well=well, frame=t0, center_y=int(cy), center_x=int(cx), ta_errors=ta,
                              ours_errors=ours, ours_uncertain=n_unc, score=ta - 2 * ours + 0.5 * min(n_unc, 2)))
    table = pd.DataFrame(found).sort_values('score', ascending=False)
    picked, rows = [], []
    for r in table.itertuples():                  # one candidate per neighbourhood
        if all(r.well != q.well or abs(r.frame - q.frame) > 2 or abs(r.center_y - q.center_y) > CROP[0]
               or abs(r.center_x - q.center_x) > CROP[1] for q in picked):
            picked.append(r)
            rows.append(r._asdict())
        if len(picked) == top:
            break
    return pd.DataFrame(rows).drop(columns='Index')


def colour_of(lineage, tracks, frames, shown):
    """Colours: GT cell -> colour (cells with a detection shown(frame, label) get distinct colours first);
    predicted track -> colour of the GT cell of its first known detection."""
    def cells_of(select):
        return sorted({lineage.cell_of[l] for f in frames for l in tracks['ours'].query('frame_index == @f').instance_id
                       if l < UNKNOWN_OFFSET and l in lineage.cell_of and select(f, l)})
    inside = cells_of(shown)
    cells = inside + [c for c in cells_of(lambda f, l: True) if c not in inside]
    cell_colour = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cells)}
    track_colour = {}
    for key, df in tracks.items():
        shown = df[df.frame_index.isin(frames)].sort_values('frame_index')
        track_colour[key] = {}
        for tid, g in shown.groupby('track_id'):
            known = [l for l in g.instance_id if l < UNKNOWN_OFFSET and l in lineage.cell_of]
            track_colour[key][tid] = cell_colour[lineage.cell_of[known[0]]] if known else UNKNOWN
    return cell_colour, track_colour


def draw(ax, image, mask, colours, box):
    y0, x0 = box
    img = image[y0:y0 + CROP[0], x0:x0 + CROP[1]].astype(float)
    lo, hi = np.percentile(img, [1, 99.5])
    rgb = np.repeat(np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)[..., None], 3, axis=2)
    m = mask[y0:y0 + CROP[0], x0:x0 + CROP[1]]
    edges = find_boundaries(m, mode='inner')
    for _ in range(EDGE_WIDTH - 1):              # thicken inwards
        edges = find_boundaries(np.where(edges, 0, m), mode='inner') | edges
    for label in np.unique(m[m > 0]):
        sel = edges & (m == label)
        rgb[sel] = matplotlib_rgb(colours.get(int(label), UNKNOWN))
    ax.imshow(rgb, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def matplotlib_rgb(hex_colour):
    return np.array([int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--well')
    parser.add_argument('--frame', type=int, help='first shown frame')
    parser.add_argument('--center', type=int, nargs=2, metavar=('ROW', 'COL'))
    args = parser.parse_args()
    style.setup()
    cands = candidates()
    if args.well:
        well, t0, (cy, cx) = args.well, args.frame, args.center
    else:
        best = cands.iloc[0]
        well, t0, cy, cx = best.well, int(best.frame), int(best.center_y), int(best.center_x)
    tracks, lineage, gt, uncertain, centre = load_well(well)
    errors = {k: incoming_errors(tracks[k], gt) for k in tracks}
    frames = list(range(t0, t0 + FRAMES))
    seq = {f: (img, msk) for f, img, msk in sequence(data.SCRIPTA, well)}
    first = tifffile.imread(seq[t0][0])
    y0 = int(np.clip(cy - CROP[0] // 2, 0, first.shape[0] - CROP[0]))
    x0 = int(np.clip(cx - CROP[1] // 2, 0, first.shape[1] - CROP[1]))
    shown = lambda f, l: (y0 - 40 <= centre[(f, l)][0] < y0 + CROP[0] + 40 and
                          x0 - 40 <= centre[(f, l)][1] < x0 + CROP[1] + 40)
    cell_colour, track_colour = colour_of(lineage, tracks, frames, shown)

    fig, axes = plt.subplots(len(ROWS), FRAMES, figsize=(style.FULL, 95 * style.MM))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.95, bottom=0.07, wspace=0.03, hspace=0.06)
    outline = [patheffects.withStroke(linewidth=1.2, foreground='black')]
    shown_errors = defaultdict(int)
    for j, f in enumerate(frames):
        image, mask = tifffile.imread(seq[f][0]), tifffile.imread(seq[f][1])
        for i, (key, title) in enumerate(ROWS):
            ax = axes[i, j]
            if key == 'gt':
                colours = {int(l): cell_colour.get(lineage.cell_of.get(int(l)), UNKNOWN)
                           for l in np.unique(mask) if l}
            else:
                df = tracks[key][tracks[key].frame_index == f]
                colours = {int(r.instance_id): track_colour[key][r.track_id] for r in df.itertuples()}
            draw(ax, image, mask, colours, (y0, x0))
            if key != 'gt' and j > 0:
                for d, kind in errors[key].items():
                    if d[0] != f or d not in centre:
                        continue
                    y, x = centre[d][0] - y0, centre[d][1] - x0
                    if 0 <= y < CROP[0] and 0 <= x < CROP[1]:
                        shown_errors[(key, kind)] += 1
                        if kind == 'false':
                            ax.plot(x, y, marker='x', ms=5, mew=1.4, color='#FF2020', path_effects=outline)
                        else:
                            ax.plot(x, y, marker='o', ms=7, mfc='none', mew=1.2, color='#FF2020', path_effects=outline)
            if key == 'ours' and j > 0:
                for (s, d), p in uncertain.items():
                    if d[0] == f:
                        y, x = centre[d][0] - y0, centre[d][1] - x0
                        if 8 <= y < CROP[0] - 8 and 8 <= x < CROP[1] - 30:
                            ax.text(x + 6, y - 6, f'p={p:.2f}', color='#FFE45C', fontsize=5.5, path_effects=outline)
            if i == 0:
                ax.set_title(f'frame {f}', fontsize=6.5, pad=2)
            if j == 0:
                ax.set_ylabel(title, fontsize=7)
    handles = [plt.Line2D([], [], ls='', marker='x', ms=5, mew=1.4, color='#FF2020', label='false link'),
               plt.Line2D([], [], ls='', marker='o', ms=6, mfc='none', mew=1.2, color='#FF2020', label='missed link'),
               plt.Line2D([], [], ls='', marker='$p$', ms=5, color='#C9A800', label='uncertain link of ours (p)')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.53, 0.0))

    lines = [f'Figure 4. Test well {well}, frames {frames[0]}-{frames[-1]}, crop rows {y0}-{y0 + CROP[0]}, '
             f'cols {x0}-{x0 + CROP[1]} (pixels).',
             'Contour colour = identity (GT cell; a predicted track takes the colour of the GT cell of its first known '
             'detection in these frames). Grey = no GT identity (label >= 50000). Markers on the incoming link of a '
             'detection: cross = false link, ring = missed GT link. p = probability of an uncertain chosen link of ours.',
             'Errors shown: ' + ', '.join(f'{k} {kind} {n}' for (k, kind), n in sorted(shown_errors.items())),
             'Candidate regions (score = Trackastra ft errors - 2 x ours errors + 0.5 x min(uncertain links of ours, 2); '
             'pass --well --frame --center ROW COL to choose):', cands.to_string(index=False)]
    style.save(fig, 'fig4_qualitative', '\n'.join(lines))


if __name__ == '__main__':
    main()
