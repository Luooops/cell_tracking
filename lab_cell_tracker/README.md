# Lab cell tracker

Tracks cell nuclei through a live-cell movie: given the segmentation masks of every frame, it decides which
detection in one frame is the same cell as which detection in the next, and stitches the answers into tracks.

This folder has the code, the trained model and the evaluation script. The 15 annotated wells we trained and
tested on are too large for the repository (6.3 GB) and are shared separately: unpack them into `data/` inside
this folder (layout below) and the numbers can be reproduced with one command.

```
python evaluate.py          # tracks the 3 test wells with the shipped model and prints the scores below
```

## Results

Three test wells that were never used for training or for tuning any setting. All methods get the same Cellpose
masks and are scored by the same code; where a method has settings, they were tuned on the 12 development wells only.

| method | perfect tracks | id switches | purity | CT | IDF1 | TRA |
|---|---|---|---|---|---|---|
| DeepCell / Caliban, pretrained | 30.0% | 291 | 0.877 | 0.193 | 0.749 | 0.927 |
| ultrack | 32.5% | 361 | 0.944 | 0.180 | 0.753 | 0.924 |
| btrack | 42.9% | 240 | 0.941 | 0.243 | 0.778 | 0.931 |
| the lab's classical tracker, best settings | 44.0% | 173 | 0.914 | 0.273 | 0.801 | 0.931 |
| trackpy | 46.5% | 194 | 0.948 | 0.292 | 0.792 | 0.933 |
| LapTrack (the LAP algorithm of TrackMate) | 47.4% | 182 | 0.945 | 0.303 | 0.797 | 0.933 |
| Trackastra, as released | 54.1% | 152 | 0.967 | 0.341 | 0.819 | 0.935 |
| Trackastra, fine-tuned on our development wells | 55.5% | 165 | 0.978 | 0.348 | 0.832 | 0.935 |
| our previous model (no appearance learning) | 59.5% | 96 | 0.961 | 0.386 | 0.854 | 0.935 |
| **this model** | **60.0%** | **89** | 0.959 | 0.393 | 0.852 | 0.935 |

Per well (perfect / id switches / purity): r16c02 62.6% / 62 / 0.963, r01c20 44.4% / 106 / 0.944,
r07c07 72.9% / 99 / 0.970. `python evaluate.py` prints exactly these.

CT, IDF1 and TRA are official Cell Tracking Challenge measures (computed with py-ctcmetrics). TRA is the
challenge's headline number, but it is driven by detection errors, and every method here gets the same detections —
so it hardly moves (0.924 to 0.935) and is not the column to compare trackers by. CT (complete tracks) and IDF1
react to linking.

How to read the comparison:

- The classical trackers and the released Trackastra used no annotation from this plate; the fine-tuned Trackastra
  and our models did. The like-for-like comparison is with the fine-tuned Trackastra: about 4.5 points more perfect
  tracks and roughly 45% fewer id switches. Trackastra keeps the purer tracks.
- Against our own previous model the gain is in id switches only (fewer in all three wells, 8% on average). The
  number of perfect tracks is the same within noise: 23 tracks are perfect only with this model, 22 only with the
  previous one. On the development wells (cross-validated) it is 61.1% / 123 against 60.1% / 133.
- For the classical trackers we tuned the distance settings only. Several of them can also use size or intensity
  in their cost; with that switched on they would probably do better than shown.
- The closest published method, CellTrack-GNN (Ben-Haim & Riklin-Raviv, ECCV 2022), is not in the table: we have not
  run it on this data.
- These three wells were looked at three times while the appearance part was being developed, so the last two rows
  are a little less independent than the others.

*Perfect track*: every detection of an annotated cell carries one predicted id, and that predicted track contains
no other annotated cell. One wrong link anywhere fails it, which is why the numbers look low next to link-level
accuracy (above 98%). *Id switches*: how often the predicted id changes along an annotated cell, summed over a
well. *Purity*: share of a predicted track that belongs to its main cell.

## Setup

Python 3.10. Install PyTorch for your machine first, then `pip install -r requirements.txt`.
A GPU is only needed for training; tracking a well takes a few seconds on a CPU.
(`LAB_TRACKER_DEVICE=cpu` forces the CPU.)

The first time a well is used its 30 frames are read from `data/` and the detections and crops are cached in
`cache/` (about a minute per well, 0.9 GB for all 15). The cache can be deleted at any time.

## What is in the folder

| | |
|---|---|
| `data.py` | reads masks and images, makes the detection table and one 64×64 crop per cell |
| `graph.py` | builds one graph per pair of frames: cells are nodes, possible links are edges |
| `model.py` | the network that scores every possible link |
| `solver.py` | turns the scores into a consistent set of links (Hungarian matching, a cell may stay unlinked) |
| `track.py` | runs the two rounds of linking over a whole movie |
| `metrics.py` | perfect tracks, id switches, purity |
| `train.py` | training |
| `evaluate.py` | scores wells with the shipped model |
| `track_division.py` | optional variant of `track.py` that lets a cell divide; read its header before using it |
| `weights/` | the trained model: three networks that differ only in their random seed |
| `data/` | the 15 wells; not in the repository, see "The data" below |
| `data_preparation/` | the script that produced `data/` from the raw annotation, for reference |

Each file starts with a plain description of its step, and most can be run on their own to print real shapes and
numbers, e.g. `python graph.py r16c02 5` or `python model.py`.

## How it works

```
masks + images ─data.py──> one row per cell (position, area, brightness) + a 64×64 picture of the cell
               ─graph.py─> for frames t and t+1: every pair of cells closer than a radius r is a candidate link
               ─model.py─> a probability for every candidate link
               ─solver.py> the best set of links in which every cell is used at most once
               ─track.py─> round 1: link neighbouring frames
                           round 2: re-join a track that ends at t with one that starts at t+2
```

**The radius r** is not a setting. It is estimated from the movie itself: about twice the largest step cells take
between frames, but never more than 1.5 times the typical distance between neighbouring cells. It is recomputed
along the movie, so it follows changes in speed and density.

**What the model looks at** for a candidate link between cell *i* and cell *j*:
- geometry: offset, distance and area ratio, all relative to r;
- competition: is *j* the closest option of *i*, and how many options does each have (kept from an earlier
  version; on its own it did not help, and we have not re-tested the final model without it);
- motion: where *i* would be if it kept moving as in the previous step, and how far *j* is from that;
- appearance: a small network turns each cell's picture into 32 numbers; the similarity of the two vectors, and how
  that similarity compares with *i*'s and *j*'s other options;
- context: two rounds of attention-based message passing let every cell see its neighbours and its candidates
  before the link is scored.

**Appearance has its own training objective.** Besides the usual "is this link right" loss, the appearance network
is asked directly: among this cell's candidates, which one is the same cell? Without that second loss the
appearance network learned nothing useful (we checked: zeroing its output changed nothing).

**A cell may stay unlinked.** The matching is solved with an extra "no partner" option for every cell. Without it
a cell whose true partner is missing (a segmentation miss) is forced onto a neighbour, and the error spreads.

**Round 2** exists because the segmentation misses about 3% of the nuclei. A missed frame cuts a track in two;
round 2 scores the pair (t, t+2) with the same model and joins the pieces if it is very sure (0.9).
The model is trained on pairs one, two and three frames apart for this reason.

Model size: 91,457 parameters. Training: 25 minutes per seed on one consumer GPU.

## Where the performance comes from

Development wells, linking neighbouring frames only, same candidate links and same solver for every row:

| a candidate link is scored by | perfect / id switches |
|---|---|
| distance only | 47.4 / 250 |
| appearance similarity only | 49.1 / 231 |
| ten hand-made shape and brightness numbers + distance, no network | 56.2 / 172 |
| appearance similarity + distance, no graph network | 58.1 / 165 |
| graph network without appearance | 59.5 / 158 |
| graph network with appearance (this model) | 60.3 / 149 |
| ... plus round 2 | 61.1 / 123 |

Appearance and distance are each worth little alone and a lot together. The graph network adds one or two points
on top, and round 2 mostly removes id switches.

## The data

`data/` holds 15 wells of one plate, 30 frames each, 2160×2160 pixels, nuclear channel. It is not part of the
repository; put the shared copy here so that the folder looks like this (or point the environment variable
`LAB_TRACKER_DATA` at wherever it is):

```
data/<well>/t000.tif … t029.tif                  raw frames
data/<well>_GT/TRA/man_track000.tif …            masks; pixel value = id of the cell
data/<well>_GT/track_id_map.csv                  our id <-> track_id in the original annotation file
data/<well>_GT/removed_polygons.csv              every annotated polygon we did not keep, with the reason
data/build_summary.json                          per-well counts
```

Twelve wells are used for development (training and all design decisions, with 3-fold cross-validation over whole
wells) and three are held out for testing: r16c02, r01c20, r07c07.

### How the data was cleaned

The annotation came as polygons with a track id, 111,155 of them. We could not use the polygons directly: about
90% were pre-labels from an older segmentation model, and in some wells they are regions about 2.5 times larger
than the nucleus; the other 10% were drawn by hand, mostly as boxes of four or five points. Area, brightness,
position and the cell's picture all depend on the outline, so we kept the annotation only for *which cell is which*
and took every outline from our fine-tuned Cellpose model. The rules, in order:

1. **Duplicate ids.** If one track id is carried by two or more polygons in the same frame, all polygons with that
   id are dropped from that frame (we cannot tell which one is right). 87 cases, 175 polygons.
2. **Segment every frame with Cellpose** (our fine-tuned model, cell-probability threshold −2, flow threshold 0.8.
   These are looser than the defaults: recall of annotated nuclei rises from 94.6% to 96.8%, and every miss is a
   hole in a track.)
3. **Match polygons to Cellpose masks, one to one.** A polygon claims the mask it overlaps most, if the overlap
   covers at least half of the mask. Hand-drawn boxes are matched afterwards with a more forgiving rule (half of
   the box or half of the mask), because they are often smaller than the cell.
4. **A matched polygon** gives its track id to the Cellpose mask. 107,946 detections with an identity.
5. **An unmatched polygon is removed** — the cell was annotated but Cellpose did not find it, so the track has a
   gap in that frame. 3,034 polygons (2,220 hand-drawn, 814 others).
6. **A Cellpose mask that no polygon claimed is kept**, with an id of 50000 or more. 5,460 of them (4.8% of all
   detections): cells the annotation missed, cells that lost their id by rule 1, debris, cells cut by the image
   border. A tracker meets these in real use, so they stay in the data; but their identity is unknown, so links
   that touch them are left out of the training loss and of the scores unless the answer is certain.
7. **Jumps.** If a track id moves more than 300 pixels between two frames it is cut there and the second part
   gets a new id. This happened twice.

After cleaning: 113,406 detections, 6,378 annotated tracks; 1,242 tracks have at least one missing frame (603
before cleaning — the difference is Cellpose misses).

### A known problem we did not fix

In 97 places the annotation gives one nucleus a new track id in the middle of the movie (same position, same
shape, a different number from one frame to the next; a quarter of them in well r14c03). The tracker links these
correctly and is scored as wrong. Counting them as one cell raises the development score from 61.1% to 62.7%.
The data here is left as annotated; the scores above are not corrected for this.

## Training your own

```
python train.py --out my_weights          # 3 seeds on the 12 development wells, ~25 min each
python evaluate.py --weights my_weights
python train.py --cv                      # 3-fold cross-validation, 9 trainings, ~2.5 h
```

Training is not bit-for-bit repeatable on a GPU. Expect a single network to move by a bit under one point of
perfect tracks from run to run; that is why three are averaged.

## Limits

- **It needs annotated tracks from the same kind of data.** The appearance part does not carry over to other
  data: on a public HeLa dataset (nuclei a third of the size, constant divisions) it gave no benefit, while the
  geometric part worked without retraining.
- **The appearance cue is worth little.** About one point of perfect tracks and 8% fewer id switches on the
  development wells; on the test wells only the id switches improved.
- **No cell division.** Links are one to one. `track_division.py` shows how the solver can allow two daughters,
  but this model was never trained on divisions.
- **Only one missed frame is bridged.** A second round for two missed frames made things worse every time we tried.
- **About a third of the tracks still contain a wrong link**, and most of what is left to gain is there, not in
  re-joining broken tracks.

## Credits

The parts of this tracker are not new. Graph networks that classify links come from multi-object tracking
(Brasó & Leal-Taixé, CVPR 2020) and were brought to cells by Ben-Haim & Riklin-Raviv (ECCV 2022), who also train an
appearance embedding by metric learning on masked crops and feed its similarity to the graph network — the
appearance part here follows their idea. Matching with a "no link" alternative and closing gaps in a second pass
go back to Jaqaman et al. (Nature Methods 2008), the algorithm behind TrackMate. Trackastra (Gallusser & Weigert,
ECCV 2024) is the strongest general-purpose tracker we compared with. Segmentation is Cellpose.
