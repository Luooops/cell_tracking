"""
lineage.py
==========
One reading of CTC lineage for every dataset: man_track.txt (L B E P per line) + the detections actually present
in the masks -> the ground-truth links between observed detections, each with its kind.

Two ways of writing "the same cell continues" occur, sometimes in the same dataset, and both mean continuation:
    same label      a label observed at frames t and t+g (g > 1 when the segmentation missed it in between;
                    scripta_data writes every gap like this, Fluo-N2DL-HeLa where the ST masks miss an observation)
    single child    a track whose parent has exactly ONE child (CTC convention for an interrupted track)
A parent with exactly TWO children divided: mother -> each daughter. Three or more children is not a valid
division under our convention; those links are kept but marked unknown (left out of supervision and scores).

Links join observed detections: from the last frame in which the earlier track is present in the masks to the first
frame in which the later one is, so g counts real frames (a mother missing in her last frame gives a g = 2 link).
Labels >= UNKNOWN_OFFSET are detections without identity: they take part in no link.

    lineage = read_lineage(path_to_man_track_txt, observations)     # observations: iterable of (frame, label)
    lineage.links                                                   # [Link, ...] in time order
    lineage.anomalies                                               # [dict(kind=..., label=..., ...), ...]
    lineage.relation((t, a), (t2, b))                               # same_cell / mother_daughter / unrelated / unknown
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

UNKNOWN_OFFSET = 50000
CONTINUATION, MOTHER_DAUGHTER, UNKNOWN = 'continuation', 'mother_daughter', 'unknown'
SAME_CELL, UNRELATED = 'same_cell', 'unrelated'


@dataclass(frozen=True)
class Link:
    """A ground-truth link src -> dst between two observed detections (frame, label); g = dst frame - src frame.
    via: 'same_label' | 'single_child' | 'two_children' | 'many_children' (how the table expresses it)."""
    src: tuple
    dst: tuple
    kind: str
    via: str

    @property
    def g(self):
        return self.dst[0] - self.src[0]


def read_track_table(path):
    """man_track.txt -> {label: (begin, end, parent)}. Raises on malformed lines and duplicate labels."""
    path = Path(path)
    table = {}
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 4:
            raise ValueError(f'{path}:{number}: expected "label begin end parent"')
        label, begin, end, parent = map(int, values)
        if label <= 0 or begin < 0 or end < begin or parent < 0 or label in table:
            raise ValueError(f'{path}:{number}: invalid or duplicate track')
        table[label] = (begin, end, parent)
    return table


class Lineage:
    def __init__(self, table, observations):
        """table: {label: (begin, end, parent)}; observations: iterable of (frame, label) present in the masks."""
        self.table = dict(table)
        self.anomalies = []
        frames = defaultdict(list)
        for frame, label in observations:
            frames[int(label)].append(int(frame))
        for label, fs in frames.items():
            if len(fs) != len(set(fs)):
                raise ValueError(f'label {label} occurs more than once in one frame')
        self.observed = {label: sorted(fs) for label, fs in frames.items()}
        self.children = defaultdict(list)
        for label, (_, _, parent) in sorted(self.table.items()):
            if parent:
                self.children[parent].append(label)
        self._check()
        self.cell_of = {}
        for label in sorted(self.table, key=lambda l: (self.table[l][0], l)):   # parents begin before children
            self.cell_of[label] = self._cell(label)
        self.links = self._links()

    @staticmethod
    def known(label):
        return 0 < label < UNKNOWN_OFFSET

    def _check(self):
        for label, fs in self.observed.items():
            if not self.known(label):
                continue
            if label not in self.table:
                self.anomalies.append(dict(kind='label_not_in_table', label=label))
                continue
            begin, end, _ = self.table[label]
            if fs[0] < begin or fs[-1] > end:
                self.anomalies.append(dict(kind='observation_outside_table', label=label, first=fs[0], last=fs[-1],
                                           begin=begin, end=end))
        for label, (begin, end, parent) in sorted(self.table.items()):
            if not self.known(label):
                continue
            fs = self.observed.get(label)
            if not fs:
                self.anomalies.append(dict(kind='unobserved_track', label=label, begin=begin, end=end))
            elif fs[0] != begin or fs[-1] != end:
                self.anomalies.append(dict(kind='unobserved_endpoint', label=label, first=fs[0], last=fs[-1],
                                           begin=begin, end=end))
            if parent and parent not in self.table:
                self.anomalies.append(dict(kind='missing_parent', label=label, parent=parent))
            elif parent and not self.known(parent):
                self.anomalies.append(dict(kind='unknown_parent', label=label, parent=parent))
        for parent, kids in sorted(self.children.items()):
            if parent not in self.table:
                continue
            end = self.table[parent][1]
            if len(kids) >= 3:
                self.anomalies.append(dict(kind='many_children', label=parent, children=kids))
            for kid in kids:
                begin = self.table[kid][0]
                if begin <= end or (len(kids) == 2 and begin != end + 1):
                    self.anomalies.append(dict(kind='child_begin_mismatch', label=kid, parent=parent,
                                               parent_end=end, child_begin=begin))

    def _cell(self, label):
        """Identity of a cell: a track and its chain of single-child continuations share the root track's label."""
        parent = self.table[label][2]
        if parent in self.table and len(self.children[parent]) == 1 and self.known(parent):
            return self.cell_of.get(parent) or self._cell(parent)
        return label

    def _links(self):
        links = []
        for label, fs in self.observed.items():
            if self.known(label) and label in self.table:
                links += [Link((a, label), (b, label), CONTINUATION, 'same_label') for a, b in zip(fs[:-1], fs[1:])]
        for parent, kids in self.children.items():
            if not self.known(parent) or not self.observed.get(parent):
                continue
            kind, via = {1: (CONTINUATION, 'single_child'), 2: (MOTHER_DAUGHTER, 'two_children')}.get(
                len(kids), (UNKNOWN, 'many_children'))
            last = self.observed[parent][-1]
            for kid in kids:
                if self.known(kid) and self.observed.get(kid) and self.observed[kid][0] > last:
                    links.append(Link((last, parent), (self.observed[kid][0], kid), kind, via))
        return sorted(links, key=lambda k: (k.src, k.dst))

    def relation(self, a, b):
        """Relation of the detections a = (frame, label) and a later b = (frame, label) as identities:
        same_cell, mother_daughter (b belongs to a daughter of a's cell; the caller decides whether a is the
        mother's last observation), unrelated, or unknown (no identity, or a parent with >= 3 children)."""
        la, lb = int(a[1]), int(b[1])
        if not (self.known(la) and self.known(lb) and la in self.table and lb in self.table):
            return UNKNOWN
        ca, cb = self.cell_of[la], self.cell_of[lb]
        if ca == cb:
            return SAME_CELL
        parent = self.table[cb][2]
        if parent in self.table and self.cell_of.get(parent) == ca:
            return MOTHER_DAUGHTER if len(self.children[parent]) == 2 else UNKNOWN
        return UNRELATED

    def summary(self):
        known = [l for l in self.table if self.known(l)]
        n_kids = Counter(len(k) for p, k in self.children.items() if self.known(p))
        gaps = sum(1 for l in known if self.observed.get(l) and
                   any(b - a > 1 for a, b in zip(self.observed[l][:-1], self.observed[l][1:])))
        return dict(tracks=len(known), unknown_identity_tracks=len(self.table) - len(known),
                    tracks_with_parent=sum(1 for l in known if self.table[l][2]),
                    divisions=n_kids.get(2, 0), single_child_continuations=n_kids.get(1, 0),
                    parents_with_many_children=sum(v for k, v in n_kids.items() if k >= 3),
                    tracks_with_internal_gap=gaps,
                    links={f'{k}/{v}': n for (k, v), n in sorted(Counter((l.kind, l.via) for l in self.links).items())},
                    anomalies=dict(Counter(a['kind'] for a in self.anomalies)))


def read_lineage(track_table, observations):
    """track_table: path to man_track.txt; observations: iterable of (frame, label) present in the masks."""
    return Lineage(read_track_table(track_table), observations)
