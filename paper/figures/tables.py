"""LaTeX tables (booktabs) -> out/tables.tex (main paper: scripta test wells, DNN, module ablations),
out/tables_suppl.tex (scripta 15 wells, division switch, HeLa, threshold sweep; the 15-well numbers are also in
figure 1) and out/table_modules_full.tex (the supplement's full module table). LNK / CT / TRA / BC are means over wells; ED (false), EA (missed), EC (wrong edge type) are AOGM terms and
the division counts are summed over wells; division recall is pooled (tp / (tp + fn)). The best value of each column
is bold (not in the module table, which compares each module with its own full model)."""
import data
import style

OURS = '\\method{}'                 # method and dataset names are macros of the paper's preamble
TEST = [('ours', OURS), ('binary', OURS + ', binary'), ('ta_ft_div', 'Trackastra ft'), ('ta_zs_div', 'Trackastra zs'),
        ('ta_ft', 'Trackastra ft$^\\ddagger$'), ('ta_zs', 'Trackastra zs$^\\ddagger$'),
        ('ctgnn', 'CellTrack-GNN'), ('classical', 'Classical$^\\ddagger$')]   # ctgnn: mean of its two training seeds
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
    # the 13 RAW264.7 videos only; test07-08 (3T3/HEK293, a cell type absent from training, which the RAW-only
    # model never tracked) are reported in the limitations
    cols = [('LNK', 'LNK', None, max, 4), ('BC', 'BC', None, max, 3), ('rec', 'Rec.', None, max, 3),
            ('fp', 'FP', None, min, 0), ('ED', 'ED', None, min, 0), ('EA', 'EA', None, min, 0)]
    rows = [(name, dnn_summary(key, data.DNN_VIDEOS)) for key, name in keys]
    return table(rows, cols, '\\textbf{Tracking with divisions} on the 13 DynamicNuclearNet (DNN) videos val17--26 '
                 'and test09--11 (RAW264.7), which no model in the table was trained on; all link with divisions '
                 'allowed. ' + OURS + ', RAW only is trained on the RAW264.7 videos alone; Trackastra zs '
                 '(\\texttt{general\\_2d}) and ctc are the released models, and \\texttt{general\\_2d} may include the '
                 'validation videos. Rec.: division recall; FP: false divisions; ED, EA: false and missed edges.',
                 'tab:dnn', colsep=3)


TAUS = [('tau0.2', 0.2), ('tau0.3', 0.3), (None, 0.4), ('tau0.5', 0.5), ('tau0.6', 0.6)]   # None: the main runs


def modules():
    """Every module with the experiment that removes it: the full model -> the model without it, on the scripta
    development wells (out of fold) and the 13 DNN videos. Retracks with TTA (gap closing, threshold) compare
    with the main runs; the cue ablations were tracked without views and compare with the same run without views
    ('base'). Training gaps: the model trained with g = 1 only (g1, g1_nogap; scripta only, retrained).
    -> (compact table for the paper: the change without - with, grouped; full table for the supplement: every
    measure as full -> without, with a note). * : paired Wilcoxon p < 0.01 over the wells / videos."""
    dev = [w for w in data.all_wells() if w not in data.TEST_WELLS]
    s = lambda k: data.scripta_wells(k, dev)
    d = lambda k: data.dnn_wells(k)
    count = lambda n: f'{int(round(n)):,}'.replace(',', '{,}')

    def mean(a, b, digits, test=True):         # (full, without) of a per-well measure: mean, or sum for counts
        f = (lambda x: x.mean()) if digits else (lambda x: x.sum())
        return dict(full=f(a), without=f(b), digits=digits,
                    star='^*' if test and data.paired(a, b)['p'] < 0.01 else '')

    def fixed(full, without, digits):           # pooled numbers, no per-well test
        return dict(full=full, without=without, digits=digits, star='')

    def arrow(v):                                # supplement: full -> without
        f = (lambda x: f'{x:.{v["digits"]}f}') if v['digits'] else count
        return f'{f(v["full"])} $\\to$ {f(v["without"])}' + ('$^*$' if v['star'] else '')

    def delta(v):                                # paper: without - with, signed; zero without a sign
        # from the values as the full table shows them, so a reader gets the same change from either table
        x = round(v['without'], v['digits']) - round(v['full'], v['digits'])
        text = f'{x:+.{v["digits"]}f}' if v['digits'] else f'{int(round(x)):+d}'
        if float(text) == 0:
            text = text.lstrip('+-')
        return f'${text}{v["star"]}$'

    best = lambda get: max((get(k) for k, _ in TAUS if k), key=lambda t: t.LNK.mean())
    area = lambda name: data.answers(name)[0]['coverage']['false_links']['mutual_info']['area']
    ece2 = lambda name: data.answers(name)[0]['calibration']['round2_solver']['ece']     # second-pass candidates
    div = lambda k, wells: data.divisions(k, wells, 'scripta')['divisions']
    fp = lambda k: d(k)['fp_div(0)'].sum()
    # (group, [(module, measure, scripta, dnn, note, in the paper's compact table)]); a cell is a value dict, a
    # fixed string, or None (not run). Velocity (no effect, at most 0.0004 LNK): supplement, figure S1.
    groups = [
        ('Relations across missed detections', [
            ('Gap closing', 'LNK $\\uparrow$', mean(s('ours_nodiv').LNK, s('nogap').LNK, 4),
             mean(d('ours').LNK, d('nogap').LNK, 4), 'adjacent frames only', True),
            ('', 'Missed edges $\\downarrow$', mean(s('ours_nodiv').AOGM_EA, s('nogap').AOGM_EA, 0),
             mean(d('ours').AOGM_EA, d('nogap').AOGM_EA, 0), '', False),
            ('Gaps 2, 3 in training$^\\dagger$', 'LNK $\\uparrow$', mean(s('ours_nodiv').LNK, s('g1').LNK, 4), None,
             'trained with $g=1$ only', True),
            ('', 'LNK$_1$ $\\uparrow$', mean(s('nogap').LNK, s('g1_nogap').LNK, 4), None, 'both without gap closing',
             False),
            ('', 'CT $\\uparrow$', mean(s('ours_nodiv').CT, s('g1').CT, 3), None, '', False),
            ('', 'ECE$_2$ $\\downarrow$', fixed(ece2('scripta_cv_oof_tta'), ece2('scripta_cv_oof_g1'), 3), None,
             'gap-closing candidates', False)]),
        ('Calibrated decisions', [
            ('Three classes$^\\dagger$', 'False div. $\\downarrow$',
             fixed(div('ours', data.TEST_WELLS), div('binary', data.TEST_WELLS), 0), fixed(fp('ours'), fp('binary'), 0),
             'binary edges; \\ds{} test wells', True),
            ('Cost-derived $\\tau=0.4$', 'LNK $\\uparrow$', mean(s('ours_nodiv').LNK, best(s).LNK, 4, test=False),
             mean(d('ours').LNK, best(d).LNK, 4, test=False), 'best $\\tau\\in\\{0.2,0.3,0.5,0.6\\}$ (\\cref{tab:tau})',
             True),
            ('\\ds{} in training$^\\dagger$', 'False div. $\\downarrow$',
             fixed(div('ours', data.all_wells()), div('raw', data.all_wells()), 0), fixed(fp('ours'), fp('raw'), 0),
             'RAW264.7 only; \\ds{}: all 15 wells', False)]),
        ('Cues and confidence', [
            ('Appearance', 'LNK $\\uparrow$', mean(s('base').LNK, s('noapp').LNK, 4),
             mean(d('base').LNK, d('noapp').LNK, 4), 'images shuffled within each frame;', True),
            ('', 'BC $\\uparrow$', 'n/a', mean(d('base')['BC(0)'], d('noapp')['BC(0)'], 3), 'without mirrored views',
             False),
            ('Mirrored views', 'MI area $\\uparrow$', fixed(area('scripta_cv_oof_tta'), area('scripta_cv_oof_base'), 3),
             fixed(area('dnn'), area('dnn_base'), 3), 'false-link coverage (\\cref{fig:confidence}b)', True)])]
    cell = lambda v, show: v if isinstance(v, str) else ('n.r.' if v is None else show(v))

    compact = ['\\begin{table}[!t]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{4pt}',
               '\\begin{tabular}{@{}llrr@{}}', '\\toprule', 'Module removed & Measure & \\ds{} & DNN \\\\', '\\midrule']
    full = ['\\begin{table*}[!t]', '\\centering', '\\small', '\\begin{tabular}{@{}lllll@{}}', '\\toprule',
            'Module removed & Measure & \\ds{} & DNN & Note \\\\', '\\midrule']
    for i, (title, rows) in enumerate(groups):
        if i:
            compact.append('\\addlinespace')
            full.append('\\addlinespace')
        compact.append(f'\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\')
        full.append(f'\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\')
        for module, measure, a, b, note, main in rows:
            if main:
                compact.append(f'{module} & {measure} & {cell(a, delta)} & {cell(b, delta)} \\\\')
            full.append(f'{module} & {measure} & {cell(a, arrow)} & {cell(b, arrow)} & {note}'.rstrip() + ' \\\\')
    compact += ['\\bottomrule', '\\end{tabular}',
                '\\caption{\\textbf{Module ablations}: change when the module is removed from the full model (without '
                '$-$ with). \\ds{}: 12 development wells, out of fold (three classes: test wells); DNN: the 13 videos '
                'of \\cref{tab:dnn} (MI area: all 15 DNN videos). $^\\dagger$Retrained. Cost-derived $\\tau$: replaced '
                'by the best tuned $\\tau$. n.r.: not run. All values, measures and definitions in '
                '\\cref{tab:modules_full}. $^*$Paired Wilcoxon $p<0.01$.}',
                '\\label{tab:modules}', '\\end{table}', '']
    full += ['\\bottomrule', '\\end{tabular}',
             '\\caption{\\textbf{Module ablation}: full model $\\to$ the same model without the module. \\ds{}: 12 '
             'development wells, out of fold, unless noted; DNN: the 13 videos of \\cref{tab:dnn} (MI area: all 15 DNN '
             'videos). $^\\dagger$Retrained without the module. n/a: \\ds{} has no divisions; n.r.: not run, the $g=1$ '
             'model was trained only in the \\ds{} cross-validation. LNK$_1$: LNK without gap closing; ECE$_2$: '
             'calibration of the gap-closing candidates; MI area: area under the false-link coverage curve. '
             '$^*$Paired Wilcoxon $p<0.01$.}',
             '\\label{tab:modules_full}', '\\end{table*}', '']
    return '\n'.join(compact), '\n'.join(full)


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
    text = table(rows, cols, '\\textbf{Linking threshold} $\\tau$ (link and gap-closing threshold; $\\tau_D$ stays 0.5). '
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
    seeds = ('ctgnn0', 'ctgnn1')
    divisions = {k: data.test_divisions(k) for k, _ in TEST if k != 'ctgnn'}
    divisions['ctgnn'] = {m: sum(data.test_divisions(s)[m] for s in seeds) / len(seeds) for m in ('divisions', 'invalid')}
    wells = lambda k: (sum(data.scripta_wells(s, data.TEST_WELLS)[[c for c, *_ in COLUMNS]] for s in seeds) / len(seeds)
                       if k == 'ctgnn' else data.scripta_wells(k, data.TEST_WELLS))   # per-well mean of the seeds
    rows = [(name, {**summary(wells(k), COLUMNS), 'div': divisions[k]['divisions']}) for k, name in TEST]
    parts.append(table(rows, COLUMNS + [DIV], '\\textbf{Test wells} of \\ds{} (r16c02, r01c20, r07c07). LNK, CT, TRA: mean over wells; ED, '
                       'EA, EC: false, missed and wrong-type edges, summed. Div.: false divisions (the ground truth '
                       'has none; CellTrack-GNN divides by post-processing and also gives '
                       f'{divisions["ctgnn"]["invalid"]:.0f} mothers three or more '
                       'daughters). Unmarked Trackastra rows link with divisions allowed, the setting of ' + OURS + '; '
                       '$^\\ddagger$divisions forbidden. ft: fine-tuned; zs: zero-shot; CellTrack-GNN: mean of two training seeds.', 'tab:test', colsep=3))
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
                       '(\\texttt{greedy}); $^\\ddagger$divisions forbidden (\\texttt{greedy\\_nodiv}).',
                       'tab:all15'), divmode(), hela(), tau()]
    compact, full = modules()
    parts += [compact, dnn()]      # the order in which sec:main and sec:divisions discuss them
    style.OUT.mkdir(parents=True, exist_ok=True)
    (style.OUT / 'tables.tex').write_text('\n'.join(parts), encoding='utf-8')
    (style.OUT / 'tables_suppl.tex').write_text('\n'.join(suppl), encoding='utf-8')
    # the supplement inputs the full module table at its own place (its section 'Module ablation')
    (style.OUT / 'table_modules_full.tex').write_text('\n'.join([parts[0], '', full]), encoding='utf-8')
    print('written', *(style.OUT / n for n in ('tables.tex', 'tables_suppl.tex', 'table_modules_full.tex')))


if __name__ == '__main__':
    main()
