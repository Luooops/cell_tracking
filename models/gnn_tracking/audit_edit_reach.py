"""
audit_edit_reach.py
===================
Stage 0 addendum: manual ID edits of a review that may have been cut off by the image strip.

A manual edit of the review page relabels a source track L in frames f0..f1 to T. The strip showed only some frames,
so an edit that ends at the strip's last frame (or starts at its first) may stop where the reviewer could not see
further, while L goes on there (gaps of at most MAX_GAP frames). Edits that end inside the strip are left alone: the
reviewer saw the next frames and stopped on purpose (e.g. a GT label swapped for two frames only).

prepare   such edits (known source labels only; unlabelled detections exist in one frame) -> one item per edit and
          direction: the frames L would be extended over -- the run of L beyond the edit, gaps <= MAX_GAP, stopping
          at a frame another edit of L covers, where L was already changed by the refined GT, or where T already
          exists -- drawn frame by frame on a strip that follows the cell (T blue, L in the proposed frames orange
          'L?', other tracks grey with their source label), and a local page: extend / keep / unsure. No model output
          is used, so every reviewed well (test wells included, labels only) can take part.
merge     decisions.json + the page's answers.json -> a new decisions file with one manual edit per 'extend' item
          (L, the proposed frames, -> T), in the review's source labels (a record of the whole decision set).
build     the 'extend' answers applied to the refined dataset itself -> a new refined dataset whose refine_log.json
          names that refined dataset as its source: each extension relabels L (unchanged there in those frames) to
          T's refined id (utils/gt_refine.refine_well, files hard-linked). Runs made on the first refined dataset are
          thus translated by these relabels only, and runs made on the original data by the chain of both.

    python -m models.gnn_tracking.audit_edit_reach prepare --dataset scripta_data --refined scripta_data_refine \\
        --review development/stage0_evaluation/handover_review_scripta_data_v6 --decisions decisions.json \\
        --output-root development/stage0_evaluation/edit_reach_review_v6
    python -m models.gnn_tracking.audit_edit_reach merge \\
        --review development/stage0_evaluation/handover_review_scripta_data_v6 --decisions decisions.json \\
        --audit development/stage0_evaluation/edit_reach_review_v6 --output decisions_extended.json
    python -m models.gnn_tracking.audit_edit_reach build --refined scripta_data_refine \
        --audit development/stage0_evaluation/edit_reach_review_v6 --target data/scripta_data_refine2
"""
import argparse
import datetime
import json
import uuid
from pathlib import Path

import numpy as np
import tifffile

from models.gnn_tracking.audit_handover import B_COLORS, UNKNOWN_OFFSET, _position, _strip, load_well
from utils.config import ctc_sequences, read_config, resolve_dataset
from utils.ctc import sequence
from utils.gt_refine import build_refined_dataset, refine_well, sha256
from utils.lineage import read_track_table
from utils.run_log import log_path, log_root, run_logged

MAX_GAP = 3
L_COLOR = (251, 140, 0)
CONTEXT = 2                     # edited frames shown next to the proposed ones


def known_edits(decisions):
    return [m for m in decisions['manual'] if isinstance(m['label'], int) and m['label'] < UNKNOWN_OFFSET]


def proposals(edits, shown, present, refined_of, refined_present):
    """-> items: edits that touch the strip's edge with frames of L beyond it, and the frames to extend over."""
    covered = {}
    for m in edits:
        covered.setdefault((m['well'], m['label']), set()).update(range(m['from_frame'], m['to_frame'] + 1))
    out = []
    for k, m in enumerate(edits):
        well, label = m['well'], m['label']
        fs = shown[m['item']]
        t_ref = refined_of[well].get((m['to_frame'], label))              # T's id in the refined GT
        for direction, start, edge in (('after', m['to_frame'], max(fs)), ('before', m['from_frame'], min(fs))):
            if start != edge:
                continue
            step, f, miss, run = (1 if direction == 'after' else -1), start, 0, []
            while True:
                f += step
                if f not in present[well] or miss > MAX_GAP:
                    break
                if f in covered[(well, label)] or (t_ref is not None and t_ref in refined_present[well][f]):
                    break
                if label in present[well][f]:
                    if refined_of[well].get((f, label)) != label:                # changed by another decision
                        break
                    run.append(f); miss = 0
                else:
                    miss += 1
            if run:
                out.append(dict(id=f"{well}:{label}:{m['from_frame']}-{m['to_frame']}:{direction}", well=well,
                                label=label, target=m['target'], target_refined=t_ref, edit_from=m['from_frame'],
                                edit_to=m['to_frame'], direction=direction, frames=sorted(run), review_item=m['item']))
    return out


def prepare(args):
    decisions = json.loads((args.review / args.decisions).read_text(encoding='utf-8'))
    review = json.loads((args.review / 'candidates.json').read_text(encoding='utf-8'))
    shown = {i['id']: i['frames_shown'] for i in review['items']}
    edits = known_edits(decisions)
    wells = sorted({m['well'] for m in edits})
    data, present, refined_of, refined_present = {}, {}, {}, {}
    for well in wells:
        data[well] = load_well(args.data_root, well)
        frames, _, masks, _ = data[well]
        present[well] = {f: set(np.unique(masks[f]).tolist()) - {0} for f in frames}
        refined_of[well], refined_present[well] = {}, {}
        for f, _, mask_path in sequence(args.refined_root, well):
            refined = tifffile.imread(mask_path)
            on = masks[f] > 0
            refined_of[well].update({(int(f), int(s)): int(r) for s, r in
                                     np.unique(np.stack([masks[f][on], refined[on]]), axis=1).T})
            refined_present[well][int(f)] = set(np.unique(refined).tolist()) - {0}
    items = proposals(edits, shown, present, refined_of, refined_present)
    out = args.output_root
    (out / 'panels').mkdir(parents=True)
    for it in items:
        frames, raws, masks, objects = data[it['well']]
        to_ref = refined_of[it['well']]
        edited = [f for f in range(it['edit_from'], it['edit_to'] + 1) if f in raws]
        edited = edited[-CONTEXT:] if it['direction'] == 'after' else edited[:CONTEXT]
        show = sorted(set(edited) | set(range(min(it['frames']), max(it['frames']) + 1)))
        show = [f for f in range(min(show) - 1, max(show) + 2) if f in raws]
        colours, centres, last = {}, {}, None
        for f in show:
            cmap = {s: (B_COLORS[0], 'T') for s in present[it['well']][f]
                    if it['target_refined'] is not None and to_ref.get((f, s)) == it['target_refined']}
            if f in it['frames']:
                cmap[it['label']] = (L_COLOR, f"{it['label']}?")
            colours[f] = cmap
            pts = [p for s in cmap if (p := _position(objects, f, s)) is not None]
            last = np.mean(pts, 0) if pts else last
            centres[f] = last
        known = [c for c in centres.values() if c is not None]
        for f in show:                                                      # frames before the cell is found
            centres[f] = centres[f] if centres[f] is not None else known[0]
        it['panel'] = f"panels/{it['well']}_{it['label']}_{it['edit_from']}_{it['edit_to']}_{it['direction']}.png"
        it['frames_shown'] = show

        def header(f, it=it):
            tag = 'edited' if it['edit_from'] <= f <= it['edit_to'] else 'extend?' if f in it['frames'] else ''
            return f'frame {f}' + (f'  ({tag})' if tag else '')
        _strip(out / it['panel'], show, centres, 60, raws, masks, {}, {}, set(), header, half=110, frame_colors=colours,
               highlight={f: L_COLOR for f in it['frames']})
    audit = dict(audit_id=uuid.uuid4().hex, review=str(args.review.resolve()), decisions=args.decisions,
                 dataset=args.dataset_name, refined=args.refined, max_gap=MAX_GAP,
                 created=datetime.datetime.now().isoformat(timespec='seconds'), items=items,
                 rule='known-label manual edits that end at the last (start at the first) frame of their strip, with the '
                      'same source label beyond it; frames proposed = its run beyond the edit (gaps <= max_gap) up to '
                      'a frame another edit of the label covers, the refined GT already changed, or the target exists')
    (out / 'items.json').write_text(json.dumps(audit, indent=1), encoding='utf-8')
    embed = json.dumps(audit, separators=(',', ':')).replace('</', '<\\/')
    (out / 'review.html').write_text(PAGE.replace('/*__DATA__*/', embed), encoding='utf-8')
    print(f"{len(known_edits(decisions))} manual edits of known tracks; {len(items)} cut off at the strip's edge "
          f"({sum(len(i['frames']) for i in items)} frames); open {out / 'review.html'}")
    return 0


def merge(args):
    decisions = json.loads((args.review / args.decisions).read_text(encoding='utf-8'))
    audit = json.loads((args.audit / 'items.json').read_text(encoding='utf-8'))
    answers = json.loads(args.answers.read_text(encoding='utf-8'))
    if answers.get('audit_id') != audit['audit_id']:
        raise ValueError('answers.json belongs to another audit (audit_id differs)')
    chosen = {k: v.get('choice') for k, v in answers['answers'].items()}
    now = datetime.datetime.now().isoformat(timespec='seconds')
    added = [dict(well=it['well'], label=it['label'], from_frame=min(it['frames']), to_frame=max(it['frames']),
                  target=it['target'], note=f"extended {it['direction']} the edit {it['edit_from']}-{it['edit_to']} "
                  f"(audit_edit_reach {audit['audit_id'][:8]})", item=it['review_item'], time=now)
             for it in audit['items'] if chosen.get(it['id']) == 'extend']
    out = dict(decisions, manual=decisions['manual'] + added,
               extended=dict(audit_id=audit['audit_id'], added=len(added), created=now,
                             counts={c: sum(v == c for v in chosen.values()) for c in ('extend', 'keep', 'unsure')},
                             unanswered=sum(it['id'] not in chosen for it in audit['items'])))
    target = args.review / args.output
    if target.exists():
        raise FileExistsError(f'{target} exists; choose a new --output name')
    target.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f"{len(added)} edits extended ({out['extended']['counts']}, unanswered {out['extended']['unanswered']}) "
          f"-> {target}")
    return 0


def build(args):
    audit = json.loads((args.audit / 'items.json').read_text(encoding='utf-8'))
    answers = json.loads(args.answers.read_text(encoding='utf-8'))
    if answers.get('audit_id') != audit['audit_id']:
        raise ValueError('answers.json belongs to another audit (audit_id differs)')
    chosen = {k: v.get('choice') for k, v in answers['answers'].items()}
    extend = [it for it in audit['items'] if chosen.get(it['id']) == 'extend']
    if any(it['target_refined'] is None for it in extend):
        raise ValueError('an extended edit has no target in the refined dataset')
    tables, relabels, relinked = {}, {}, []
    for well in sorted({it['well'] for it in extend}):
        observed = {}
        for f, _, mask_path in sequence(args.refined_root, well):
            for label in np.unique(tifffile.imread(mask_path)):
                observed.setdefault(int(label), []).append(f)
        moves = [(it['label'], min(it['frames']), max(it['frames']), it['target_refined'], it['id'])
                 for it in extend if it['well'] == well]
        table = read_track_table(Path(args.refined_root) / f'{well}_GT' / 'TRA' / 'man_track.txt')
        # a track's child is linked from its last detection: the link follows that detection, to the target if the
        # extension moves it, else it stays with the track (refine_well refuses edits of tracks with children, so
        # the links are taken out for it and put back)
        relink = {}
        for x, f1, f2, target, _ in moves:
            for c, (b, e, p) in list(table.items()):
                if p == x:
                    table[c] = (b, e, 0); relink[c] = target if max(observed.get(x, [-1])) <= f2 else x
        tables[well], relabels[well] = refine_well(table, observed, [], [], moves)
        for c, target in relink.items():
            b, e, _ = tables[well][c]
            tables[well][c] = (b, e, target)
            relinked.append(dict(well=well, child=c, parent=target))
    modes = build_refined_dataset(args.refined_root, args.target, ctc_sequences(args.refined_root), tables, relabels)
    log = dict(source=str(Path(args.refined_root).resolve()), target=str(args.target.resolve()),
               created=datetime.datetime.now().isoformat(timespec='seconds'), audit=str(args.audit.resolve()),
               audit_id=audit['audit_id'], items_sha256=sha256(args.audit / 'items.json'),
               answers_sha256=sha256(args.answers), rule=audit['rule'],
               counts={c: sum(v == c for v in chosen.values()) for c in ('extend', 'keep', 'unsure')},
               change="manual edits of the source review cut off by its image strip, extended where the reviewer "
                      "confirmed it: the label's masks in the given frames get the target's label (in this dataset's "
                      "source, the refined dataset). Images and all other masks are identical to the source.",
               files=modes,
               extended=[dict(well=it['well'], track=it['label'], frames=it['frames'], target=it['target_refined'],
                              item=it['id']) for it in extend],
               relinked=relinked,
               relabels=[dict(well=w, label=x, from_frame=f1, to_frame=f2, new_label=n) for w in sorted(relabels)
                         for x, f1, f2, n in relabels[w]])
    (args.target / 'refine_log.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
    print(f"{len(extend)} edits extended in {len(tables)} wells; files {modes}; saved {args.target}")
    if modes['hardlink']:
        print('note: images and masks are hard links to the source files; never write into them')
    return 0


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>Edit reach</title>
<style>
body{font:14px system-ui,sans-serif;margin:0;background:#f6f7f9;color:#222}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:8px 16px;display:flex;gap:12px;align-items:center;z-index:2}
main{padding:12px 16px}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 12px}
.panel{overflow-x:auto}.panel img{display:block;max-width:none}
.choices{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
button{font:inherit;padding:6px 10px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
button.on{background:#1e88e5;color:#fff;border-color:#1e88e5}
.meta{color:#333;margin:4px 0 8px}.help{color:#666;font-size:12px}
textarea{width:100%;min-height:40px;font:inherit}
</style></head><body>
<header><b>手动修改的延伸 · 确认</b><span id="count"></span><button id="export">导出 answers.json</button>
<span class="help">键盘：1 接上 · 2 保持 · u 不确定 · ← → 翻页</span></header>
<main><div class="card">
<div class="meta" id="meta"></div>
<div class="panel"><img id="img" alt="panel"></div>
<div class="help">蓝色 T = 修改后的目标轨迹 T（按第一轮修正后的 GT）；橙色 “L?” = 原始标签 L 在修改范围之外、可能被截断的那些帧（橙框）；灰色数字 = 其他轨迹的原始标签。每帧的裁剪框跟着细胞走。问题：橙色的 L? 是不是 T 这个细胞的延续？</div>
<div class="choices" id="choices"></div>
<textarea id="note" placeholder="备注（可选）"></textarea>
<div class="choices"><button id="prev">← 上一个</button><button id="next">下一个 →</button><span id="pos"></span></div>
</div></main>
<script id="data" type="application/json">/*__DATA__*/</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById('data').textContent), items = data.items;
  const KEY = 'edit-reach-' + data.audit_id, $ = id => document.getElementById(id);
  const OPTIONS = [['extend', '1：接上（橙色这些帧也是 T）'], ['keep', '2：保持（不是 T，不改）'], ['unsure', 'u：不确定']];
  let answers = {}; try { answers = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
  let idx = 0;
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {} };
  function choose(v) { const it = items[idx]; answers[it.id] = Object.assign(answers[it.id] || {}, { choice: v }); save(); render(); }
  function render() {
    const it = items[idx], a = answers[it.id] || {};
    const dir = it.direction === 'after' ? '之后' : '之前';
    $('meta').innerHTML = `孔板 ${it.well} · 第一轮你把原始轨迹 <b>${it.label}</b> 的第 ${it.edit_from}–${it.edit_to} 帧改成了 <b>${it.target}</b>` +
      (it.target_refined !== null && String(it.target_refined) !== String(it.target) ? `（修正后的编号 ${it.target_refined}）` : '') +
      `，但那张图只显示到修改的边缘。原始轨迹 ${it.label} 在${dir}的第 ${it.frames.join(', ')} 帧还在（橙框）。这些帧要不要也改成 ${it.target}？`;
    $('img').src = it.panel;
    const box = $('choices'); box.innerHTML = '';
    OPTIONS.forEach(([v, text]) => { const b = document.createElement('button'); b.textContent = text; if (a.choice === v) b.className = 'on'; b.onclick = () => choose(v); box.appendChild(b); });
    $('note').value = a.note || '';
    $('pos').textContent = `第 ${idx + 1} / ${items.length} 个`;
    $('count').textContent = `已答 ${items.filter(i => (answers[i.id] || {}).choice).length} / ${items.length}`;
  }
  $('note').oninput = e => { const it = items[idx]; answers[it.id] = Object.assign(answers[it.id] || {}, { note: e.target.value }); save(); };
  $('prev').onclick = () => { idx = Math.max(0, idx - 1); render(); };
  $('next').onclick = () => { idx = Math.min(items.length - 1, idx + 1); render(); };
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'TEXTAREA') return;
    if (e.key === '1') choose('extend'); if (e.key === '2') choose('keep'); if (e.key === 'u') choose('unsure');
    if (e.key === 'ArrowLeft') $('prev').click(); if (e.key === 'ArrowRight') $('next').click();
  });
  $('export').onclick = () => { const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify({ audit_id: data.audit_id, exported: new Date().toISOString(), answers }, null, 1)], { type: 'application/json' }));
    a.download = 'answers.json'; a.click(); };
  render();
})();
</script></body></html>
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'merge', 'build'):
        p = sub.add_parser(name)
        p.add_argument('--config', type=Path, help='Default: project config.json')
        if name != 'build':
            p.add_argument('--review', type=Path, required=True, help='The review folder (candidates.json, decisions)')
            p.add_argument('--decisions', default='decisions.json', help='Decisions file inside --review')
        if name == 'build':
            p.add_argument('--refined', required=True, help='Refined dataset the extensions are applied to')
            p.add_argument('--audit', type=Path, required=True, help='The prepare output folder')
            p.add_argument('--answers', type=Path, help='Default: <audit>/answers.json')
            p.add_argument('--target', type=Path, required=True, help='New dataset folder')
        elif name == 'prepare':
            p.add_argument('--dataset', help='Source dataset of the review (its labels)')
            p.add_argument('--data-root', type=Path, help='Explicit CTC root; overrides the dataset root')
            p.add_argument('--refined', required=True, help='Refined dataset built from these decisions')
            p.add_argument('--output-root', type=Path, required=True, help='New directory for the audit')
        else:
            p.add_argument('--audit', type=Path, required=True, help='The prepare output folder')
            p.add_argument('--answers', type=Path, help='Default: <audit>/answers.json')
            p.add_argument('--output', required=True, help='New decisions file name inside --review')
    args = parser.parse_args(argv)
    cfg, base = read_config(args.config)
    if args.command == 'merge':
        args.answers = args.answers or args.audit / 'answers.json'
        return merge(args)
    if args.command == 'build':
        args.answers = args.answers or args.audit / 'answers.json'
        try:
            args.refined_root = resolve_dataset(cfg, base, args.refined, None)['root']
        except FileNotFoundError as exc:
            parser.error(str(exc))
        if args.target.exists():
            parser.error(f'target already exists: {args.target}; select a new directory')
        log = log_path(log_root(args.audit, base / cfg['output_root']), 'gnn_tracking',
                       f'{args.target.resolve().name}_edit_reach_build')
        return run_logged(log, build, args, argv=argv)
    try:
        spec = resolve_dataset(cfg, base, args.dataset, args.data_root)
        args.refined_root = resolve_dataset(cfg, base, args.refined, None)['root']
    except FileNotFoundError as exc:
        parser.error(str(exc))
    args.data_root, args.dataset_name = spec['root'], spec['name']
    if args.output_root.exists():
        parser.error(f'output-root already exists: {args.output_root}; select a new directory')
    log = log_path(log_root(args.output_root, base / cfg['output_root']), 'gnn_tracking',
                   f'{args.output_root.resolve().name}_edit_reach_{spec["name"]}')
    return run_logged(log, prepare, args, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
