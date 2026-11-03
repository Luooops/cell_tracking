"""LaTeX tables (booktabs) -> out/tables.tex (main paper: scripta test wells, scripta 15 wells, division switch, DNN)
and out/tables_suppl.tex (HeLa). LNK / CT / TRA / BC are means over wells; ED (false), EA (missed), EC (wrong edge
type) are AOGM terms and the division counts are summed over wells; division recall is pooled (tp / (tp + fn)).
The best value of each column is bold."""
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


def table(rows, columns, caption, label, lead=('Method',), colsep=4, star=False):
    """rows: [(name, {column: value})] -> booktabs tabular, best of each column bold. lead: headers of the leading
    text columns; a name holds one cell per lead header, joined by ' & '. colsep: \\tabcolsep in pt."""
    best = {c: pick(v[c] for _, v in rows) for c, _, _, pick, _ in columns}
    env = 'table*' if star else 'table'
    out = [f'\\begin{{{env}}}[t]', '\\centering', '\\small', f'\\setlength{{\\tabcolsep}}{{{colsep}pt}}',
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
    out = ['\\begin{table*}[t]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{4pt}',
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
                       f'{divisions["ctgnn1"]["invalid"]} mothers three or more daughters. ' + OURS + ' is never '
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
    parts.append(table(rows, cols15 + [DIV], 'All 15 \\ds{} wells: 12 development wells out-of-fold (3 folds) and the '
                       '3 test wells with the final models. Div.: false divisions. ' + OURS + ' is higher than all '
                       'four Trackastra variants in LNK and CT (Wilcoxon, largest '
                       f'$p = {tex_p(worst["LNK"])}$ / ${tex_p(worst["CT"])}$). Unmarked: divisions allowed '
                       '(\\texttt{greedy}); $^\\ddagger$told that there are no divisions (\\texttt{greedy\\_nodiv}).',
                       'tab:all15'))
    parts.append(divmode())
    parts.append(dnn())
    style.OUT.mkdir(parents=True, exist_ok=True)
    (style.OUT / 'tables.tex').write_text('\n'.join(parts), encoding='utf-8')
    (style.OUT / 'tables_suppl.tex').write_text(parts[0] + '\n\n' + hela(), encoding='utf-8')
    print('written', style.OUT / 'tables.tex', style.OUT / 'tables_suppl.tex')


if __name__ == '__main__':
    main()
