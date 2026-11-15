"""LaTeX tables (booktabs) -> out/tables.tex (main paper: scripta test wells, DNN, module ablations),
out/tables_suppl.tex (scripta 15 wells, division switch, HeLa, threshold sweep; the 15-well numbers are also in
figure 1) and out/table_modules_full.tex (the supplement's full module table). LNK / CT / TRA / BC are means over wells; ED (false), EA (missed), EC (wrong edge type) are AOGM terms and
the division counts are summed over wells; division recall is pooled (tp / (tp + fn)). The best value of each column
is bold on green (\\best, defined in the paper's preamble), the rows of ablated variants of the method excluded (not
in the module table, which compares each module with its own full model)."""
import data
import style

OURS = '\\method{}'                 # method and dataset names are macros of the paper's preamble
ALLOWED = 'Divisions allowed, as for ' + OURS
FORBIDDEN = 'Divisions forbidden$^\\ddagger$, an assumption unavailable to ' + OURS
ABLATION = 'Ablation of ' + OURS + ' (not ranked)'
# (key, name) rows and (None, group title) rows; ctgnn: mean of its two training seeds
TEST = [(None, ALLOWED), ('ours', OURS), ('ta_ft_div', 'Trackastra ft'), ('ta_zs_div', 'Trackastra zs'),
        ('ctgnn', 'CellTrack-GNN'),
        (None, FORBIDDEN), ('ta_ft', 'Trackastra ft$^\\ddagger$'), ('ta_zs', 'Trackastra zs$^\\ddagger$'),
        ('classical', 'Classical$^\\ddagger$'),
        (None, ABLATION), ('binary', OURS + ', binary')]
SCORES_ERRORS = [('CTC scores', 2, 4), ('Edge errors', 5, 7)]       # header groups over COLUMNS
# the short names (ft fine-tuned, zs zero-shot, s0/s1 training seeds) keep the tables inside one column
COLUMNS = [('LNK', 'LNK', 'mean', max, 3), ('CT', 'CT', 'mean', max, 3), ('TRA', 'TRA', 'mean', max, 3),
           ('AOGM_ED', 'ED', 'sum', min, 0), ('AOGM_EA', 'EA', 'sum', min, 0), ('AOGM_EC', 'EC', 'sum', min, 0)]
DIV = ('div', 'Div.', None, min, 0)      # false divisions (data.division_events)


BEST_NOTE = ('$\\uparrow$: higher is better, $\\downarrow$: lower is better; the best value of each column is bold on '
             'green')


def table(rows, columns, caption, label, lead=('Method',), colsep=4, star=False, bold=True, ablation=(), top=()):
    """rows: [(name, {column: value})] -> booktabs tabular. A row (None, title) starts a group of rows under an
    italic title. The header marks each column with its better direction (max: up, min: down); the best value of
    each column (as printed, so ties are all marked) is \\best (bold on green, preamble) unless bold=False, and
    rows named in ablation (variants of the method itself) are not candidates for it. lead: headers of the leading
    text columns; a name holds one cell per lead header, joined by ' & '. top: a header row above the column names,
    [(title, first column, last column)] (1-based, counting the lead columns). colsep: \\tabcolsep in pt."""
    fmt = lambda x, digits: f'{x:.{digits}f}' if digits else f'{int(round(x))}'
    data_rows = [(n, v) for n, v in rows if n is not None]
    best = {c: fmt(pick(float(fmt(v[c], d)) for n, v in data_rows if n not in ablation), d) if bold else None
            for c, _, _, pick, d in columns}
    width = len(lead) + len(columns)
    env = 'table*' if star else 'table'
    out = [f'\\begin{{{env}}}[!t]', '\\centering', '\\small', f'\\setlength{{\\tabcolsep}}{{{colsep}pt}}',
           '\\begin{tabular}{' + 'l' * len(lead) + 'r' * len(columns) + '}', '\\toprule']
    if top:
        out.append(' & '.join([''] * (top[0][1] - 1) + [f'\\multicolumn{{{b - a + 1}}}{{c}}{{{t}}}' for t, a, b in top])
                   + ' \\\\')
        out.append(''.join(f'\\cmidrule(lr){{{a}-{b}}}' for _, a, b in top))
    out += [' & '.join(lead) + ' & ' + ' & '.join(h + ('$\\uparrow$' if pick is max else '$\\downarrow$')
                                                   for _, h, _, pick, _ in columns) + ' \\\\', '\\midrule']
    first = True
    for name, v in rows:
        if name is None:
            if not first:
                out.append('\\addlinespace')
            out.append(f'\\multicolumn{{{width}}}{{@{{}}l}}{{\\emph{{{v}}}}} \\\\')
        else:
            cells = []
            for c, _, _, _, digits in columns:
                text = fmt(v[c], digits)
                cells.append(f'\\best{{{text}}}' if text == best[c] and name not in ablation else text)
            out.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')
        first = False
    out += ['\\bottomrule', '\\end{tabular}', f'\\caption{{{caption}}}', f'\\label{{{label}}}', f'\\end{{{env}}}', '']
    return '\n'.join(out)


def divmode():
    """Test wells, each method with divisions forbidden and allowed (ours: its two tracking modes; Trackastra:
    greedy_nodiv and greedy), and the false divisions with divisions allowed."""
    keys = [('ours', 'ours_nodiv', OURS), ('ta_ft_div', 'ta_ft', 'Trackastra ft'), ('ta_zs_div', 'ta_zs', 'Trackastra zs'),
            (None, None, ABLATION), ('binary', 'binary_nodiv', OURS + ', binary')]
    cols = [('LNK0', 'LNK', None, max, 4), ('CT0', 'CT', None, max, 3), ('LNK1', 'LNK', None, max, 4),
            ('CT1', 'CT', None, max, 3), ('div', 'Div.', None, min, 0)]
    rows = []
    for div, nodiv, name in keys:
        if div is None:
            rows.append((None, name))
            continue
        told, free = (data.scripta_wells(k, data.TEST_WELLS) for k in (nodiv, div))
        rows.append((name, dict(LNK0=told.LNK.mean(), CT0=told.CT.mean(), LNK1=free.LNK.mean(), CT1=free.CT.mean(),
                                div=data.test_divisions(div)['divisions'])))
    return table(rows, cols, '\\textbf{Divisions forbidden or allowed} (\\ds{} test wells, no divisions in the ground '
                 'truth). ' + BEST_NOTE + ' (ablation excluded). Divisions forbidden: our tracking mode without the '
                 'division step; Trackastra \\texttt{greedy\\_nodiv}. Divisions allowed: our division mode; Trackastra '
                 '\\texttt{greedy}. Div.: false divisions with divisions allowed. ' + OURS + ' gives the same result '
                 'either way, so it needs no division switch.', 'tab:divmode', ablation={OURS + ', binary'},
                 top=[('Divisions forbidden', 2, 3), ('Divisions allowed', 4, 6)])


def dnn_summary(key, videos):
    t = data.dnn_wells(key, videos)
    tp, fn = t['tp_div(0)'].sum(), t['fn_div(0)'].sum()
    return dict(LNK=t.LNK.mean(), BC=t['BC(0)'].mean(), rec=tp / (tp + fn), fp=t['fp_div(0)'].sum(), ED=t.AOGM_ED.sum(),
                EA=t.AOGM_EA.sum())


def dnn():
    keys = [('ours', OURS), ('ta_ft_div', 'Trackastra ft'), ('ta_zs_div', 'Trackastra zs'), ('ta_zs_ctc', 'Trackastra ctc'),
            (None, ABLATION), ('binary', OURS + ', binary'), ('raw', OURS + ', RAW only')]
    # the 13 RAW264.7 videos only; test07-08 (3T3/HEK293, a cell type absent from training, which the RAW-only
    # model never tracked) are reported in the limitations
    cols = [('LNK', 'LNK', None, max, 4), ('BC', 'BC', None, max, 3), ('rec', 'Rec.', None, max, 3),
            ('fp', 'FP', None, min, 0), ('ED', 'ED', None, min, 0), ('EA', 'EA', None, min, 0)]
    rows = [(None, name) if key is None else (name, dnn_summary(key, data.DNN_VIDEOS)) for key, name in keys]
    return table(rows, cols, '\\textbf{Tracking with divisions} on the 13 DynamicNuclearNet (DNN) videos val17--26 '
                 'and test09--11 (RAW264.7), which no model in the table was trained on; all link with divisions '
                 'allowed. ' + BEST_NOTE + ' (ablations excluded). ' + OURS + ', RAW only is trained on the RAW264.7 videos alone; Trackastra zs '
                 '(\\texttt{general\\_2d}) and ctc are the released models, and \\texttt{general\\_2d} may include the '
                 'validation videos. Rec.: division recall; FP: false divisions; ED, EA: false and missed edges.',
                 'tab:dnn', colsep=3, ablation={OURS + ', binary', OURS + ', RAW only'})


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
             fixed(area('dnn'), area('dnn_base'), 3), 'false-link coverage (\\cref{fig:ranking}a)', True)])]
    cell = lambda v, show: v if isinstance(v, str) else ('n.r.' if v is None else show(v))

    compact = ['\\begin{table}[!t]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{4pt}',
               '\\begin{tabular}{@{}llrr@{}}', '\\toprule', 'Module removed & Metric & \\ds{} & DNN \\\\', '\\midrule']
    full = ['\\begin{table*}[!t]', '\\centering', '\\small', '\\begin{tabular}{@{}lllll@{}}', '\\toprule',
            'Module removed & Metric & \\ds{} & DNN & Note \\\\', '\\midrule']
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
                '\\caption{\\textbf{Module ablations}: change when the module is removed (without $-$ with). \\ds{}: '
                '12 development wells, out of fold (three classes: test wells); DNN: 13 videos (MI area: all 15). '
                '$^\\dagger$Retrained. $\\tau$: replaced by the best tuned one. n.r.: not run. Full values in '
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
    return table(rows, cols, '\\textbf{Linking threshold} $\\tau$ (link and gap-closing threshold; $\\tau_D$ stays 0.5). '
                 '\\ds{}: 12 development wells, out of fold; DNN: 13 videos. $\\uparrow$: higher is better, '
                 '$\\downarrow$: lower is better. $\\tau=0.4$ follows from the LNK costs '
                 '(\\cref{sec:decisions}). Paired Wilcoxon of \\ds{} LNK against $\\tau=0.4$: ' +
                 ', '.join(f'{t:.1f}: $p={p[t]:.2f}$' if p[t] >= 0.01 else f'{t:.1f}: $p={tex_p(p[t])}$' for t in sorted(p))
                 + '.', 'tab:tau', lead=('$\\tau$',), colsep=3, bold=False, top=[('\\ds{}', 2, 5), ('DNN', 6, 9)])


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
                 '(except $^\\dagger$). ' + BEST_NOTE + ' (ablation excluded). ' + OURS + ' and Trackastra ft: trained on \\ds{} and the DNN RAW264.7 '
                 'videos. Trackastra zs: \\texttt{general\\_2d}. All link with divisions allowed. BC and division '
                 'recall at tolerance 0; FP: false divisions. $^\\dagger$The \\texttt{ctc} model, trained on all '
                 'CTC ground truth including these sequences.', 'tab:hela', lead=('Method', 'Seq.'), colsep=3,
                 ablation={f'{OURS}, binary & {seq}' for seq in ('01', '02')})


def main():
    parts = ['% Generated by paper/figures/tables.py from development/final_results_v2; do not edit by hand.', '']
    seeds = ('ctgnn0', 'ctgnn1')
    divisions = {k: data.test_divisions(k) for k, _ in TEST if k not in (None, 'ctgnn')}
    divisions['ctgnn'] = {m: sum(data.test_divisions(s)[m] for s in seeds) / len(seeds) for m in ('divisions', 'invalid')}
    wells = lambda k: (sum(data.scripta_wells(s, data.TEST_WELLS)[[c for c, *_ in COLUMNS]] for s in seeds) / len(seeds)
                       if k == 'ctgnn' else data.scripta_wells(k, data.TEST_WELLS))   # per-well mean of the seeds
    rows = [(None, name) if k is None else (name, {**summary(wells(k), COLUMNS), 'div': divisions[k]['divisions']})
            for k, name in TEST]
    # stated in the supplement's baseline details (no longer in the caption): parents with >= 3 children
    print(f'CellTrack-GNN, test wells: {divisions["ctgnn"]["invalid"]:.0f} parents with three or more children '
          '(mean of the two seeds)')
    parts.append(table(rows, COLUMNS + [DIV], '\\textbf{Test wells} of \\ds{} (r16c02, r01c20, r07c07; the ground '
                       'truth has no divisions). ' + BEST_NOTE + ' (ablation excluded). LNK, CT, TRA: mean over '
                       'wells; ED, EA, EC: false, missed and wrong-type edges, summed over wells; Div.: false '
                       'divisions. $^\\ddagger$Trackastra is told that no cell divides (\\texttt{greedy\\_nodiv}); the '
                       'classical tracker never divides. '
                       'ft: fine-tuned; zs: zero-shot; CellTrack-GNN: mean of two training seeds; binary: ' + OURS +
                       ' with a binary edge classifier (\\cref{tab:modules}).', 'tab:test', colsep=3,
                       ablation={OURS + ', binary'}, top=SCORES_ERRORS))
    keys = [(None, ALLOWED), ('ours', OURS), ('ta_ft_div', 'Trackastra fine-tuned'), ('ta_zs_div', 'Trackastra zero-shot'),
            (None, FORBIDDEN), ('ta_ft', 'Trackastra fine-tuned$^\\ddagger$'),
            ('ta_zs', 'Trackastra zero-shot$^\\ddagger$')]
    cols15 = [c for c in COLUMNS if c[0] != 'AOGM_EC']
    wells = data.all_wells()
    rows = [(None, name) if k is None else
            (name, {**summary(data.scripta_wells(k, wells), cols15), 'div': data.scripta_divisions(k, wells)['divisions']})
            for k, name in keys]
    worst = {m: max(data.paired(data.scripta_wells('ours', wells)[m], data.scripta_wells(k, wells)[m])['p']
                    for k, _ in keys if k not in (None, 'ours')) for m in ('LNK', 'CT')}
    suppl = [parts[0], '', table(rows, cols15 + [DIV], 'All 15 \\ds{} wells: 12 development wells out-of-fold (3 folds) and the '
                       '3 test wells with the final models. ' + BEST_NOTE + '. ED, EA: false and missed edges; Div.: '
                       'false divisions. ' + OURS + ' is higher than all four Trackastra variants in LNK and CT '
                       f'(Wilcoxon, largest $p = {tex_p(worst["LNK"])}$ / ${tex_p(worst["CT"])}$). Divisions allowed: '
                       '\\texttt{greedy}; $^\\ddagger$divisions forbidden: \\texttt{greedy\\_nodiv}.',
                       'tab:all15', top=[('CTC scores', 2, 4), ('Edge errors', 5, 6)]), divmode(), hela(), tau()]
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
