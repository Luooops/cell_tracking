"""
audit_confident.py
==================
Stage 1 diagnostic: look at the tracker's confident errors before choosing how to improve it.

A confident false link is a round-1 link the tracker chose although the ground truth says the two detections are
different cells, with every ensemble member above 0.9. They are sorted by what the ground truth says around them:
    mixup        the source's true successor was among its candidates; the tracker picked another cell
    a_ends_b_old the source's track ends at t, the target belongs to a track that existed before t + 1
    a_gap_b_new  the source's track continues later (missed frame), the target starts a track at t + 1
    a_ends_b_new the source's track ends at t, the target starts a track at t + 1 (hand-over pattern)
    other
and a stratified random sample (every class at least min(5, its size), the rest in proportion) is drawn. Each item gets
one image strip, frames t-3 .. t+4 with one fixed crop: every detection of the source's cell outlined red 'A',
every detection of the target's cell blue 'B' (cells as utils/lineage.py reads the GT: single-child continuations
included), other tracks grey with their id, unlabelled detections grey '?'. The local page review.html asks which
of these it is:
    1 the GT is wrong, the tracker is right      2 ambiguous (cells pressed together, merged / split detection)
    3 tracker wrong, visible over several frames 4 tracker wrong, not visible even over several frames      5 other
The answers only steer the next step; they are not used for training or scoring.

    python -m models.gnn_tracking.audit_confident --dataset scripta_data_refine \
        --run development/stage1_calibration/oof_cv_v003_lnk/oof/gnn_tracking \
        --output-root development/stage1_calibration/confident_errors_cv_v003
"""
import argparse
import datetime
import json
import uuid
from pathlib import Path

import numpy as np

from models.gnn_tracking.audit_handover import A_COLOR, B_COLORS, _position, _strip, load_well as load_images
from models.gnn_tracking.evaluate_answers import load_well
from utils.calibration import sigmoid
from utils.config import read_config, resolve_dataset
from utils.lineage import SAME_CELL, UNRELATED, Lineage, read_track_table
from utils.run_log import log_path, log_root, run_logged

CONFIDENT = 0.9
CLASSES = ('mixup', 'a_ends_b_old', 'a_gap_b_new', 'a_ends_b_new', 'other')
CONTEXT = 3


def confident_false_links(edges, relation, table):
    """-> [dict(t, a, b, q, cls)] for round-1 links chosen although unrelated, every member above CONFIDENT."""
    e = edges
    r1 = e['round'] == 1
    q = sigmoid(e['logits'])
    true_src = {(int(t), int(a)) for t, a, ok in zip(e['src_frame'][r1], e['src_label'][r1], relation[r1] == SAME_CELL) if ok}
    sel = r1 & e['chosen'] & (relation == UNRELATED) & (q > CONFIDENT).all(1)
    out = []
    for t, a, b, qq in zip(e['src_frame'][sel], e['src_label'][sel], e['dst_label'][sel], q[sel]):
        t, a, b = int(t), int(a), int(b)
        a_ends = a in table and table[a][1] == t
        b_starts = b in table and table[b][0] == t + 1
        if (t, a) in true_src:
            cls = 'mixup'
        elif a_ends and b_starts:
            cls = 'a_ends_b_new'
        elif a_ends:
            cls = 'a_ends_b_old'
        elif b_starts:
            cls = 'a_gap_b_new'
        else:
            cls = 'other'
        out.append(dict(t=t, a=a, b=b, q=[round(float(x), 3) for x in qq], cls=cls))
    return out


def confident_missed_links(edges, relation):
    """-> [dict(t, a, b, q, cls='missed', model_b)] for round-1 same-cell edges the tracker left out, every member
    below 1 - CONFIDENT; model_b = the detection the tracker linked a to instead (None if none)."""
    e = edges
    r1 = e['round'] == 1
    q = sigmoid(e['logits'])
    linked = {(int(t), int(a)): int(b) for t, a, b in zip(e['src_frame'][r1 & e['chosen']], e['src_label'][r1 & e['chosen']],
                                                         e['dst_label'][r1 & e['chosen']])}
    sel = r1 & ~e['chosen'] & (relation == SAME_CELL) & (q < 1 - CONFIDENT).all(1)
    return [dict(t=int(t), a=int(a), b=int(b), q=[round(float(x), 3) for x in qq], cls='missed',
                 model_b=linked.get((int(t), int(a))))
            for t, a, b, qq in zip(e['src_frame'][sel], e['src_label'][sel], e['dst_label'][sel], q[sel])]


def stratified_sample(items, n, seed):
    """Every class at least min(5, its size), the rest in proportion to class size; fixed order by a seeded shuffle."""
    rng = np.random.default_rng(seed)
    by = {c: [i for i in items if i['cls'] == c] for c in CLASSES}
    take = {c: min(5, len(v)) for c, v in by.items()}
    rest = max(n - sum(take.values()), 0)
    total = sum(len(v) - take[c] for c, v in by.items())
    for c, v in by.items():
        if total:
            take[c] = min(len(v), take[c] + int(round(rest * (len(v) - take[c]) / total)))
    picked = []
    for c, v in by.items():
        idx = rng.permutation(len(v))[:take[c]]
        picked += [v[i] for i in sorted(idx)]
    order = rng.permutation(len(picked))
    return [picked[i] for i in order]


def render(item, well, raws, masks, objects, lineage, radius, path, blue=None):
    """Strip of frames t-3 .. t+4: the source's GT cell red 'A', the GT cell of `blue` (default: the target b)
    blue 'B'. -> (frames shown, {label: frames visible}, {unlabelled label: tag}) as audit_handover._strip."""
    t, a = item['t'], item['a']
    b = item['b'] if blue is None else blue
    cell_a = lineage.cell_of.get(a, a)
    members_a = [l for l, c in lineage.cell_of.items() if c == cell_a] or [a]
    if b is None:
        members_b = []
    else:
        cell_b = lineage.cell_of.get(b, b)
        members_b = [l for l, c in lineage.cell_of.items() if c == cell_b and c != cell_a] or ([b] if b != a else [])
    return draw(t, members_a, members_b, raws, masks, objects, radius, path)


def draw(t, members_a, members_b, raws, masks, objects, radius, path):
    """The strip of render() for given label sets ('A' red, 'B' blue), in the label space of `masks`."""
    show = [f for f in range(t - CONTEXT, t + CONTEXT + 2) if f in raws]
    pts = np.array([p for f in show for l in members_a + members_b if (p := _position(objects, f, l)) is not None])
    lo, hi = pts.min(0), pts.max(0)
    half = int(np.clip(max(radius * 1.2, (hi - lo).max() / 2 + 60), 48, 320))
    centres = {f: (lo + hi) / 2 for f in show}
    colors = {l: A_COLOR for l in members_a}
    colors.update({l: B_COLORS[0] for l in members_b})
    names = {l: 'A' for l in members_a}
    names.update({l: 'B' for l in members_b})

    def header(f):
        return f'frame {f}' + ('  (t: A)' if f == t else '  (t+1: B)' if f == t + 1 else f'  (t{f - t:+d})')
    return _strip(path, show, centres, radius, raws, masks, colors, names, set(), header, half=half)


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>Confident errors</title>
<style>
body{font:14px system-ui,sans-serif;margin:0;background:#f6f7f9;color:#222}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:8px 16px;display:flex;gap:12px;align-items:center;z-index:2}
main{padding:12px 16px;max-width:1500px}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 12px}
.panel{overflow-x:auto}.panel img{display:block;max-width:none}
.choices{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
button{font:inherit;padding:6px 10px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
button.on{background:#1e88e5;color:#fff;border-color:#1e88e5}
.meta{color:#555;margin:4px 0 8px}.help{color:#777;font-size:12px}
textarea{width:100%;min-height:40px;font:inherit}
</style></head><body>
<header><b>自信的错连 · 抽查</b><span id="count"></span><button id="export">导出 answers.json</button>
<span class="help">键盘：1–5 选择，← → 翻页</span></header>
<main><div class="card">
<div class="meta" id="meta"></div>
<div class="panel"><img id="img" alt="panel"></div>
<div class="help">红色 A = GT 中源细胞（A 在第 t 帧被模型连出去）的所有检测；蓝色 B = GT 中目标细胞（模型在第 t+1 帧连到的那个检测）的所有检测；灰色数字 = 其他轨迹；灰色 ? = 没有身份的检测。所有帧用同一个裁剪框，位置变化就是真实的移动。</div>
<div class="choices" id="choices"></div>
<textarea id="note" placeholder="备注（可选）"></textarea>
<div class="choices"><button id="prev">← 上一个</button><button id="next">下一个 →</button><span id="pos"></span></div>
</div></main>
<script id="data" type="application/json">/*__DATA__*/</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById('data').textContent), items = data.items;
  const KEY = 'confident-errors-' + data.audit_id, $ = id => document.getElementById(id);
  const OPTIONS = [['gt_wrong', '1：GT 标错，模型是对的'], ['ambiguous', '2：模棱两可（细胞贴在一起 / 检测合并或拆分）'],
                   ['model_context', '3：模型错了，前后几帧能看出来'], ['model_hard', '4：模型错了，前后几帧也看不出来'], ['other', '5：其他 / 不确定']];
  const CLS = { mixup: 'A 真正的下一帧也在候选里，模型选了别的细胞', a_ends_b_old: 'GT：A 在第 t 帧结束，B 是一条更早就存在的轨迹',
                a_gap_b_new: 'GT：A 之后漏检、稍后才出现，B 在第 t+1 帧新开始', a_ends_b_new: 'GT：A 在第 t 帧结束，B 在第 t+1 帧新开始', other: '其他' };
  let answers = {}; try { answers = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
  let idx = 0;
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {} };
  function render() {
    const it = items[idx], a = answers[it.id] || {};
    $('meta').textContent = `${it.id} · 孔板 ${it.well} · 第 t = ${it.t} 帧 · 模型把 A（轨迹 ${it.a}）连到了 B（轨迹 ${it.b}），三个成员的概率 ${it.q.join(' / ')} · ${CLS[it.cls]}`;
    $('img').src = it.panel;
    const box = $('choices'); box.innerHTML = '';
    OPTIONS.forEach(([v, text]) => { const b = document.createElement('button'); b.textContent = text;
      if (a.choice === v) b.classList.add('on'); b.onclick = () => choose(v); box.appendChild(b); });
    $('note').value = a.note || '';
    $('pos').textContent = `第 ${idx + 1} / ${items.length} 个`;
    $('count').textContent = `已看 ${items.filter(x => (answers[x.id] || {}).choice).length} / ${items.length}`;
  }
  function choose(v) { const it = items[idx]; answers[it.id] = Object.assign({}, answers[it.id], { choice: v, time: new Date().toISOString() });
    save(); if (idx < items.length - 1) idx++; render(); }
  $('note').oninput = e => { const it = items[idx]; answers[it.id] = Object.assign({}, answers[it.id], { note: e.target.value }); save(); };
  $('prev').onclick = () => { if (idx > 0) { idx--; render(); } };
  $('next').onclick = () => { if (idx < items.length - 1) { idx++; render(); } };
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'TEXTAREA') return;
    if (e.key >= '1' && e.key <= '5') choose(OPTIONS[+e.key - 1][0]);
    if (e.key === 'ArrowLeft') $('prev').click();
    if (e.key === 'ArrowRight') $('next').click();
  });
  $('export').onclick = () => { const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify({ audit_id: data.audit_id, exported: new Date().toISOString(), answers }, null, 1)], { type: 'application/json' }));
    a.download = 'answers.json'; a.click(); };
  render();
})();
</script></body></html>"""


def run(args, wells):
    out = args.output_root
    (out / 'panels').mkdir(parents=True)
    found = []
    for well in wells:
        part = load_well(args.run, args.data_root, well)
        table = read_track_table(Path(args.data_root) / f'{well}_GT' / 'TRA' / 'man_track.txt')
        found += [dict(well=well, **x) for x in confident_false_links(part['edges'], part['relation'], table)]
    counts = {c: sum(i['cls'] == c for i in found) for c in CLASSES}
    print(f'{len(found)} confident false links: {counts}', flush=True)
    sample = stratified_sample(found, args.n, args.seed)
    by_well = {}
    for i, item in enumerate(sample):
        item['id'] = f'e{i + 1:03d}'
        by_well.setdefault(item['well'], []).append(item)
    for well, todo in sorted(by_well.items()):
        frames, raws, masks, objects = load_images(args.data_root, well)
        lineage = Lineage(read_track_table(Path(args.data_root) / f'{well}_GT' / 'TRA' / 'man_track.txt'), ())
        edges = load_well(args.run, args.data_root, well)['edges']
        for item in todo:
            sel = (edges['round'] == 1) & (edges['src_frame'] == item['t'])
            radius = float(edges['gate'][sel][0]) if sel.any() else 100.0
            item['panel'] = f"panels/{item['id']}_{well}_t{item['t']}.png"
            item['frames'] = render(item, well, raws, masks, objects, lineage, radius, out / item['panel'])[0]
        print(f'{well}: {len(todo)} panels', flush=True)
    data = dict(audit_id=uuid.uuid4().hex, run=str(Path(args.run).resolve()), dataset=args.dataset_name,
                created=datetime.datetime.now().isoformat(timespec='seconds'), seed=args.seed, counts=counts,
                rule=f'round-1 chosen links between different GT cells, every member above {CONFIDENT}; stratified sample',
                items=sample)
    (out / 'items.json').write_text(json.dumps(data, indent=1), encoding='utf-8')
    (out / 'review.html').write_text(PAGE.replace('/*__DATA__*/', json.dumps(data).replace('</', '<\\/')), encoding='utf-8')
    print(f'{len(sample)} items; open {out / "review.html"}')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, help='Default: project config.json')
    parser.add_argument('--dataset', help='Dataset name from the config, or a folder under data_dir')
    parser.add_argument('--data-root', type=Path, help='Explicit CTC root; overrides the dataset root')
    parser.add_argument('--run', type=Path, required=True, help='Folder with <well>/edges.npz + instance_tracks.csv')
    parser.add_argument('--n', type=int, default=60, help='Sample size')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-root', type=Path, required=True, help='New directory for the audit')
    args = parser.parse_args(argv)
    cfg, base = read_config(args.config)
    try:
        spec = resolve_dataset(cfg, base, args.dataset, args.data_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    args.data_root, args.dataset_name = spec['root'], spec['name']
    wells = sorted(p.parent.name for p in args.run.glob('*/edges.npz'))
    if not wells:
        parser.error('no wells with edges.npz under --run')
    if set(wells) & set(spec['held_out']):
        parser.error(f'test wells must not be audited here: {sorted(set(wells) & set(spec["held_out"]))}')
    if args.output_root.exists():
        parser.error(f'output-root already exists: {args.output_root}; select a new directory')
    log = log_path(log_root(args.output_root, base / cfg['output_root']), 'gnn_tracking',
                   f'{args.output_root.resolve().name}_audit_confident_{spec["name"]}')
    return run_logged(log, run, args, wells, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
