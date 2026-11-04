"""LaTeX tables (booktabs) -> out/tables.tex (main paper: scripta test wells, DNN, module ablations) and
out/tables_suppl.tex (scripta 15 wells, division switch, HeLa, threshold sweep; the 15-well numbers are also in
figure 1). LNK / CT / TRA / BC are means over wells; ED (false), EA (missed), EC (wrong edge type) are AOGM terms and
the division counts are summed over wells; division recall is pooled (tp / (tp + fn)). The best value of each column
is bold (not in the module table, which compares each module with its own full model)."""
import data
import style

OURS = '\\method{}'                 # method and dataset names are macros of the paper's preamble
TEST = [('ours', OURS), ('binary', OURS + ', binary'), ('ta_ft_div', 'Trackastra ft'), ('ta_zs_div', 'Trackastra zs'),
        ('ta_ft', 'Trackastra ft$^\\ddagger$'), ('ta_zs', 'Trackastra zs$^\\ddagger$'),
        ('ctgnn0', 'CellTrack-GNN s0'), ('ctgnn1', 'CellTrack-GNN s1'), ('classical', 'Classical$^\\ddagger$')]
# the short names (ft fine-tuned, zs zero-shot, s0/s1 training seeds) keep the tables inside one column
COLUMNS = [('LNK', 'LNK', 'mean', max, 4), ('CT', 'CT', 'mean', max, 3), ('TRA', 'TRA', 'mean', max, 4),
           ('AOGM_ED', 'ED', 'sum', min, 0), ('AOGM_EA', 'EA', 'sum', min, 0), ('AOGM_EC', 'EC', 'sum', min, 0)]
DIV = ('div', 'Div.', None, min, 0)      # false divisions (data.division_events)


def table(rows, columns, caption, label, lead=('Method',), colsep=4, star=False, bold=True):
    """rows: [(name, {column: value})] -> booktabs tabular, best of each column bold (unless bold=False). lead:
    headers of the leading text columns; a name holds one cell per lead header, joined by ' & '. colsep: \\tabcolsep
    in pt."""
    best = {c: pick(v[c] for _, v in rows) if bold else None for c, _, _, pick, _ in columns}
    env = 'table*' if star else 'table'
    out = [f'\\begin{{{env}}}[!t]', '\\centering', '\\small', f'\\setlength{{\\tabcolsep}}{{{colsep}pt}}',
           '\\begin{tabular}{' + 'l' * len(lead) + 'r' * len(columns) + '}', '\\toprule',
           ' & '.join(lead) + ' & ' + ' & '.join(h for _, h, *_ in columns) + ' \\\\', '\\midrule']
    for name, v in rows:
        cells = []
        for c, _, _, _, digits in columns:
            text = f'{v[c]:.{digits}f}' if digits else f'{int(round(v[c]))}'
            cells.append(f'\\textbf{{{text}}}' if v[c] == best[c] else text)
        out.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabular}', f'\\caption{{{caption}}}', f'\\label{{{label}}}', f'\\end{{{env}}}', '']
    return '\n'.join(out)


def grouped(text, old_header, groups):
    """Replace the header line by a two-level header: groups = [(title, first column, last column)] (1-based, counting
    the lead columns)."""
    top = ' & '.join([''] + [f'\\multicolumn{{{b - a + 1}}}{{c}}{{{t}}}' for t, a, b in groups]) + ' \\\\'
    rules = ''.join(f'\\cmidrule(lr){{{a}-{b}}}' for _, a, b in groups)
    return text.replace(old_header, f'{top}\n{rules}\n{old_header}', 1)


def divmode():
    """Test wells, each method linked without and with divisions (ours: its two tracking modes; Trackastra:
    greedy_nodiv and greedy), and the false divisions with divisions allowed."""
    keys = [('ours', 'ours_nodiv', OURS), ('binary', 'binary_nodiv', OURS + ', binary'),
            ('ta_ft_div', 'ta_ft', 'Trackastra ft'), ('ta_zs_div', 'ta_zs', 'Trackastra zs')]
    cols = [('LNK0', 'LNK', None, max, 4), ('CT0', 'CT', None, max, 3), ('LNK1', 'LNK', None, max, 4),
            ('CT1', 'CT', None, max, 3), ('div', 'Div.', None, min, 0)]
    rows = []
    for div, nodiv, name in keys:
        told, free = (data.scripta_wells(k, data.TEST_WELLS) for k in (nodiv, div))
        rows.append((name, dict(LNK0=told.LNK.mean(), CT0=told.CT.mean(), LNK1=free.LNK.mean(), CT1=free.CT.mean(),
                                div=data.test_divisions(div)['divisions'])))
    text = table(rows, cols, '\\textbf{Telling the tracker that there are no divisions} (\\ds{} test wells, no divisions '
                 'in the ground truth). No divisions: our no-division tracking mode; Trackastra '
                 '\\texttt{greedy\\_nodiv}. Divisions allowed: our division mode; Trackastra \\texttt{greedy}. Div.: '
                 'false divisions with divisions allowed. ' + OURS + ' gives the same result either way, so it '
                 'needs no division switch.', 'tab:divmode')
    return grouped(text, 'Method & LNK & CT & LNK & CT & Div. \\\\', [('No divisions', 2, 3), ('Divisions allowed', 4, 6)])


def dnn_summary(key, videos):
    t = data.dnn_wells(key, videos)
    tp, fn = t['tp_div(0)'].sum(), t['fn_div(0)'].sum()
    return dict(LNK=t.LNK.mean(), BC=t['BC(0)'].mean(), rec=tp / (tp + fn), fp=t['fp_div(0)'].sum(), ED=t.AOGM_ED.sum(),
                EA=t.AOGM_EA.sum())


def dnn():
    keys = [('ours', OURS), ('binary', OURS + ', binary'), ('raw', OURS + ', RAW only'), ('ta_ft_div', 'Trackastra ft'),
            ('ta_zs_div', 'Trackastra zs'), ('ta_zs_ctc', 'Trackastra ctc')]
    cols = [(f'{m}{s}', h, None, pick, d) for s in ('13', 'off')
            for m, h, pick, d in (('LNK', 'LNK', max, 4), ('BC', 'BC', max, 3), ('rec', 'Rec.', max, 3), ('fp', 'FP', min, 0))]
    cols.insert(4, ('ED13', 'ED', None, min, 0))
    cols.insert(5, ('EA13', 'EA', None, min, 0))
    rows = []
    for key, name in keys:
        a = dnn_summary(key, data.DNN_VIDEOS)
        values = {f'{m}13': a[m] for m in ('LNK', 'BC', 'rec', 'fp', 'ED', 'EA')}
        try:                    # the RAW-only model never tracked the 3T3/HEK293 videos test07-08
            b = dnn_summary(key, data.DNN_OFFICIAL)
            values.update({f'{m}off': b[m] for m in ('LNK', 'BC', 'rec', 'fp')})
        except KeyError:
            values.update({f'{m}off': float('nan') for m in ('LNK', 'BC', 'rec', 'fp')})
        rows.append((name, values))
    best = {c: pick(v[c] for _, v in rows if v[c] == v[c]) for c, _, _, pick, _ in cols}
    out = ['\\begin{table*}[!t]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{4pt}',
           '\\begin{tabular}{l' + 'r' * len(cols) + '}', '\\toprule',
           ' & \\multicolumn{6}{c}{13 videos (val17--26, test09--11)} & \\multicolumn{4}{c}{Official test videos '
           '(test07--11)} \\\\', '\\cmidrule(lr){2-7}\\cmidrule(lr){8-11}',
           'Method & ' + ' & '.join(h for _, h, *_ in cols) + ' \\\\', '\\midrule']
    for name, v in rows:
        cells = []
        for c, _, _, _, digits in cols:
            if v[c] != v[c]:
                cells.append('--')
                continue
            text = f'{v[c]:.{digits}f}' if digits else f'{int(round(v[c]))}'
            cells.append(f'\\textbf{{{text}}}' if v[c] == best[c] else text)
        out.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabular}',
            '\\caption{\\textbf{Tracking with divisions} on DynamicNuclearNet (DNN) videos that no model in the table '
            'was trained on. ' + OURS + ' and Trackastra ft are trained on \\ds{} and the RAW264.7 training videos; '
            + OURS + ', RAW only on the RAW264.7 videos alone. Trackastra zs (\\texttt{general\\_2d}) and ctc are the '
            'released models; \\texttt{general\\_2d} includes the DNN training videos and may include its validation '
            'videos (val17--26), which can only favour it. All link with divisions allowed. BC at tolerance 0; Rec.: '
            'division recall, pooled; FP: false divisions; ED, EA: false and missed edges. test07--08 show '
            '3T3/HEK293 cells, a cell line absent from our training data (not run for the RAW-only model).}',
            '\\label{tab:dnn}', '\\end{table*}', '']
    return '\n'.join(out)


TAUS = [('tau0.2', 0.2), ('tau0.3', 0.3), (None, 0.4), ('tau0.5', 0.5), ('tau0.6', 0.6)]   # None: the main runs


def modules():
    """Every module with the experiment that removes it: the full model -> the model without it, on the scripta
    development wells (out of fold) and the 13 DNN videos. Retracks with TTA (cross-frame check, threshold) compare
    with the main runs; the cue ablations were tracked without views and compare with the same run without views
    ('base'). * : paired Wilcoxon p < 0.01 over the wells / videos."""
    dev = [w for w in data.all_wells() if w not in data.TEST_WELLS]
    s = lambda k: data.scripta_wells(k, dev)
    d = lambda k: data.dnn_wells(k)

    def change(a, b, digits, test=True):
        star = '$^*$' if test and data.paired(a, b)['p'] < 0.01 else ''
        f = (lambda x: f'{x:.{digits}f}') if digits else (lambda x: f'{int(round(x))}')
        mean = (lambda x: x.mean()) if digits else (lambda x: x.sum())
        return f'{f(mean(a))} $\\to$ {f(mean(b))}{star}'

    best = lambda get: max((get(k) for k, _ in TAUS if k), key=lambda t: t.LNK.mean())
    area = lambda name: data.answers(name)[0]['coverage']['false_links']['mutual_info']['area']
    div = lambda k, wells, dataset: data.divisions(k, wells, dataset)['divisions']
    rows = [
        ('Cross-frame check', 'LNK', change(s('ours_nodiv').LNK, s('nogap').LNK, 4), change(d('ours').LNK, d('nogap').LNK, 4)),
        ('', 'EA', change(s('ours_nodiv').AOGM_EA, s('nogap').AOGM_EA, 0), change(d('ours').AOGM_EA, d('nogap').AOGM_EA, 0)),
        ('Appearance', 'LNK', change(s('base').LNK, s('noapp').LNK, 4), change(d('base').LNK, d('noapp').LNK, 4)),
        ('', 'BC', '--', change(d('base')['BC(0)'], d('noapp')['BC(0)'], 3)),
        ('Velocity', 'LNK', change(s('base').LNK, s('novel').LNK, 4), change(d('base').LNK, d('novel').LNK, 4)),
        ('Three classes$^a$', 'Div.', f'{div("ours", data.TEST_WELLS, "scripta")} $\\to$ '
         f'{div("binary", data.TEST_WELLS, "scripta")}',
         f'{int(d("ours")["fp_div(0)"].sum())} $\\to$ {int(d("binary")["fp_div(0)"].sum())}'),
        ('\\ds{} in training$^b$', 'Div.', f'{div("ours", data.all_wells(), "scripta")} $\\to$ '
         f'{div("raw", data.all_wells(), "scripta")}',
         f'{int(d("ours")["fp_div(0)"].sum())} $\\to$ {int(d("raw")["fp_div(0)"].sum())}'),
        ('Mirrored views', 'MI area', f'{area("scripta_cv_oof_tta"):.3f} $\\to$ {area("scripta_cv_oof_base"):.3f}',
         f'{area("dnn"):.3f} $\\to$ {area("dnn_base"):.3f}'),
        ('$\\tau=0.4$$^c$', 'LNK', change(s('ours_nodiv').LNK, best(s).LNK, 4), change(d('ours').LNK, best(d).LNK, 4)),
    ]
    out = ['\\begin{table}[!t]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{3pt}',
           '\\begin{tabular}{llll}', '\\toprule', 'Module & Measure & \\ds{} & DNN \\\\', '\\midrule']
    out += [' & '.join(r) + ' \\\\' for r in rows]
    out += ['\\bottomrule', '\\end{tabular}',
            '\\caption{\\textbf{What each module contributes}: the full model $\\to$ the same model without the module '
            '(\\ds{}: 12 development wells, out of fold; DNN: the 13 videos of \\cref{tab:dnn}). Nothing is retrained '
            'except in $^a$ and $^b$. Cross-frame check: adjacent frames only. Appearance: cell images shuffled '
            'within each frame; velocity: no velocity input (both compared without views). Div.: false divisions; '
            'MI area: area under the false-link coverage curve (\\cref{fig:confidence}b). $^a$Binary edges (linked or '
            'not), \\ds{} test wells. $^b$Trained on RAW264.7 alone (\\ds{}: all 15 wells). $^c$Best LNK over '
            '$\\tau\\in\\{0.2,0.3,0.5,0.6\\}$ (\\cref{tab:tau}). $^*$Paired Wilcoxon $p<0.01$.}',
            '\\label{tab:modules}', '\\end{table}', '']
    return '\n'.join(out)


def tau():
    """Threshold sweep: link and gap threshold tau, the rest unchanged (retracks with views; 0.4 = the main runs)."""
    dev = [w for w in data.all_wells() if w not in data.TEST_WELLS]
    cols = [('sLNK', 'LNK', None, max, 4), ('sCT', 'CT', None, max, 3), ('sED', 'ED', None, min, 0),
            ('sEA', 'EA', None, min, 0), ('dLNK', 'LNK', None, max, 4), ('dED', 'ED', None, min, 0),
            ('dEA', 'EA', None, min, 0), ('dBC', 'BC', None, max, 3)]
    rows = []
    for key, t in TAUS:
        a, b = data.scripta_wells(key or 'ours_nodiv', dev), data.dnn_wells(key or 'ours')
        rows.append((f'{t:.1f}', dict(sLNK=a.LNK.mean(), sCT=a.CT.mean(), sED=a.AOGM_ED.sum(), sEA=a.AOGM_EA.sum(),
                                      dLNK=b.LNK.mean(), dED=b.AOGM_ED.sum(), dEA=b.AOGM_EA.sum(),
                                      dBC=b['BC(0)'].mean())))
    ref = data.scripta_wells('ours_nodiv', dev).LNK
    p = {t: data.paired(data.scripta_wells(k, dev).LNK, ref)['p'] for k, t in TAUS if k}
    text = table(rows, cols, '\\textbf{Linking threshold} $\\tau$ (link and cross-frame threshold; $\\tau_D$ stays 0.5). '
                 '\\ds{}: 12 development wells, out of fold; DNN: 13 videos. $\\tau=0.4$ follows from the LNK costs '
                 '(\\cref{sec:decisions}). Paired Wilcoxon of \\ds{} LNK against $\\tau=0.4$: ' +
                 ', '.join(f'{t:.1f}: $p={p[t]:.2f}$' if p[t] >= 0.01 else f'{t:.1f}: $p={tex_p(p[t])}$' for t in sorted(p))
                 + '.', 'tab:tau', lead=('$\\tau$',), colsep=3, bold=False)
    return grouped(text, '$\\tau$ & LNK & CT & ED & EA & LNK & ED & EA & BC \\\\', [('\\ds{}', 2, 5), ('DNN', 6, 9)])


def tex_p(p):
    mantissa, exponent = f'{p:.1e}'.split('e')
    return f'{mantissa}\\times10^{{{int(exponent)}}}'


def summary(t, columns):
    return {c: (t[c].mean() if how == 'mean' else t[c].sum()) for c, _, how, _, _ in columns}


def hela():
    hela_cols = [('LNK', 'LNK', 'mean', max, 3), ('CT', 'CT', 'mean', max, 3), ('BC(0)', 'BC', 'mean', max, 3),
                 ('div_recall', 'Recall', 'mean', max, 3), ('fp_div(0)', 'FP', 'sum', min, 0)]
    rows = []
    for k, name in (('ours', OURS), ('binary', OURS + ', binary'), ('ta_ft_div', 'Trackastra ft'),
                    ('ta_zs_div', 'Trackastra zs'), ('ta_zs_ctc', 'Trackastra ctc$^\\dagger$')):
        t = data.hela_wells(k)
        for seq in t.index:
            rows.append((f'{name} & {seq}', {c: t.loc[seq, c] for c, *_ in hela_cols}))
    return table(rows, hela_cols, 'Fluo-N2DL-HeLa, per sequence (Seq.), a cell line none of the models was trained on '
                 '(except $^\\dagger$). ' + OURS + ' and Trackastra ft: trained on \\ds{} and the DNN RAW264.7 '
                 'videos. Trackastra zs: \\texttt{general\\_2d}. All link with divisions allowed. BC and division '
                 'recall at tolerance 0; FP: false divisions. $^\\dagger$The \\texttt{ctc} model, trained on all '
                 'CTC ground truth including these sequences.', 'tab:hela', lead=('Method', 'Seq.'), colsep=3)


def main():
    parts = ['% Generated by paper/figures/tables.py from development/final_results_v2; do not edit by hand.', '']
    divisions = {k: data.test_divisions(k) for k, _ in TEST}
    rows = [(name, {**summary(data.scripta_wells(k, data.TEST_WELLS), COLUMNS), 'div': divisions[k]['divisions']})
            for k, name in TEST]
    parts.append(table(rows, COLUMNS + [DIV], 'Test wells (r16c02, r01c20, r07c07) of \\ds{}. CTC measures on fixed '
                       'detections with the project ground truth (not comparable with the CTC leaderboard). '
                       'LNK, CT, TRA: mean over wells; ED, EA, EC: AOGM terms (false, missed, wrong-type edges) '
                       'summed over wells. Div.: false divisions (a mother with exactly two daughters; the ground '
                       f'truth has none); CellTrack-GNN also gives {divisions["ctgnn0"]["invalid"]} / '
                       f'{divisions["ctgnn1"]["invalid"]} mothers three or more daughters. CellTrack-GNN has no '
                       'learned division model: its post-processing makes a track that starts in frame $t$ the '
                       'daughter of the nearest track that ended in frame $t-1$. ' + OURS + ' is never '
                       'told whether a dataset has divisions; unmarked Trackastra rows link with divisions allowed '
                       '(\\texttt{greedy}), the same setting. $^\\ddagger$Told that there are no divisions (Trackastra '
                       '\\texttt{greedy\\_nodiv}; the classical tracker never divides). ft: fine-tuned; zs: zero-shot; '
                       's0, s1: training seeds.', 'tab:test', colsep=3))
    keys = [('ours', OURS), ('ta_ft_div', 'Trackastra fine-tuned'), ('ta_zs_div', 'Trackastra zero-shot'),
            ('ta_ft', 'Trackastra fine-tuned$^\\ddagger$'), ('ta_zs', 'Trackastra zero-shot$^\\ddagger$')]
    cols15 = [c for c in COLUMNS if c[0] != 'AOGM_EC']
    wells = data.all_wells()
    rows = [(name, {**summary(data.scripta_wells(k, wells), cols15), 'div': data.scripta_divisions(k, wells)['divisions']})
            for k, name in keys]
    worst = {m: max(data.paired(data.scripta_wells('ours', wells)[m], data.scripta_wells(k, wells)[m])['p']
                    for k, _ in keys[1:]) for m in ('LNK', 'CT')}
    suppl = [parts[0], '', table(rows, cols15 + [DIV], 'All 15 \\ds{} wells: 12 development wells out-of-fold (3 folds) and the '
                       '3 test wells with the final models. Div.: false divisions. ' + OURS + ' is higher than all '
                       'four Trackastra variants in LNK and CT (Wilcoxon, largest '
                       f'$p = {tex_p(worst["LNK"])}$ / ${tex_p(worst["CT"])}$). Unmarked: divisions allowed '
                       '(\\texttt{greedy}); $^\\ddagger$told that there are no divisions (\\texttt{greedy\\_nodiv}).',
                       'tab:all15'), divmode(), hela(), tau()]
    parts += [dnn(), modules()]
    style.OUT.mkdir(parents=True, exist_ok=True)
    (style.OUT / 'tables.tex').write_text('\n'.join(parts), encoding='utf-8')
    (style.OUT / 'tables_suppl.tex').write_text('\n'.join(suppl), encoding='utf-8')
    print('written', style.OUT / 'tables.tex', style.OUT / 'tables_suppl.tex')


if __name__ == '__main__':
    main()
