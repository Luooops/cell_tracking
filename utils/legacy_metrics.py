"""Explicit-denominator track recovery metrics and pastel summary figures."""
from collections import defaultdict
import csv


def ratio(n, d):
    return n / d if d else None


def evaluate_recovery(rows, tracks):
    groups, ownership = defaultdict(list), defaultdict(set)
    for r in rows:
        if r['gt_track_id'] and not r.get('gt_id_conflict', False):
            groups[(r['sequence'], r['gt_track_id'])].append(r)
            if r['pred_track_id'] and not r['ambiguous']:
                ownership[(r['sequence'], r['pred_track_id'])].add(r['gt_track_id'])
    links = []
    for track in tracks:
        history = sorted(groups[(track['sequence'], track['gt_track_id'])], key=lambda r: r['time_index'])
        matched = [r for r in history if r['pred_track_id']]
        ids = {r['pred_track_id'] for r in matched}
        mixed = any(len(ownership[(track['sequence'], pid)]) > 1 for pid in ids)
        ambiguous = any(r['ambiguous'] for r in history)
        missing = any(not r['prediction_available'] for r in history)
        complete = len(matched) == len(history) and len(ids) == 1 and not mixed and not ambiguous and not missing
        status = ('single_observation' if len(history) < 2 else 'missing_prediction' if missing else
                  'needs_review' if ambiguous else 'recovered' if complete else 'not_recovered')
        track.update(recovery_eligible=len(history) >= 2, recovery_status=status,
                     fully_recovered=complete and len(history) >= 2,
                     has_ambiguous_observations=ambiguous, has_missing_predictions=missing,
                     prediction_identity_mixed=mixed,
                     reliable_coverage=sum(bool(r['pred_track_id']) and not r['ambiguous'] for r in history)/len(history))
        for a, b in zip(history, history[1:]):
            if b['time_index'] - a['time_index'] != 1:
                continue
            available = a['prediction_available'] and b['prediction_available']
            uncertain = a['ambiguous'] or b['ambiguous']
            both = bool(a['pred_track_id'] and b['pred_track_id'])
            clean = available and not uncertain and both
            correct = clean and a['pred_track_id'] == b['pred_track_id']
            links.append(dict(sequence=a['sequence'], gt_track_id=a['gt_track_id'],
                previous_image=a['image_name'], current_image=b['image_name'],
                predictions_available=available, ambiguous=uncertain,
                both_reliably_matched=clean, recovered=correct,
                previous_pred_id=a['pred_track_id'], current_pred_id=b['pred_track_id']))
    eligible = [t for t in tracks if t['recovery_eligible']]
    available_links = [l for l in links if l['predictions_available']]
    clean_links = [l for l in available_links if l['both_reliably_matched']]
    decidable = [l for l in available_links if not l['ambiguous']]
    correct = sum(l['recovered'] for l in available_links)
    metric_rows = []
    def add(name, n, d):
        metric_rows.append(dict(metric=name, numerator=n, denominator=d, value=ratio(n, d)))
    add('position_coverage', sum(bool(r['pred_track_id']) for r in rows), len(rows))
    add('reliable_position_coverage', sum(bool(r['pred_track_id']) and not r['ambiguous'] for r in rows), len(rows))
    add('complete_gt_track_recovery', sum(t['fully_recovered'] for t in eligible), len(eligible))
    add('conditional_identity_retention', correct, len(clean_links))
    add('gt_link_recovery_conservative', correct, len(available_links))
    add('gt_link_recovery_decidable', correct, len(decidable))
    add('ambiguous_link_fraction', sum(l['ambiguous'] for l in available_links), len(available_links))
    add('track_review_fraction', sum(t['recovery_status'] == 'needs_review' for t in eligible), len(eligible))
    add('observed_prediction_mixing_fraction', sum(len(ids) > 1 for ids in ownership.values()), len(ownership))
    bins = []
    for label, lo, hi in [('2-9', 2, 9), ('10-19', 10, 19), ('20+', 20, float('inf'))]:
        subset = [t for t in eligible if lo <= t['gt_observations'] <= hi]
        n = sum(t['fully_recovered'] for t in subset)
        bins.append(dict(length_group=label, recovered=n, total=len(subset), rate=ratio(n, len(subset))))
    counts = dict(eligible_tracks=len(eligible), single_observation_tracks=len(tracks)-len(eligible),
        missing_prediction_tracks=sum(t['has_missing_predictions'] for t in eligible),
        total_gt_adjacent_links=len(links), excluded_missing_prediction_links=len(links)-len(available_links))
    mixing = [dict(sequence=seq, pred_track_id=pid, gt_ids=';'.join(sorted(ids)), gt_count=len(ids))
              for (seq, pid), ids in sorted(ownership.items())]
    return metric_rows, bins, links, mixing, counts


def save_table(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export_recovery(output, rows, tracks):
    metrics, bins, links, mixing, counts = evaluate_recovery(rows, tracks)
    save_table(output/'metrics.csv', metrics, ['metric', 'numerator', 'denominator', 'value'])
    save_table(output/'recovery_by_length.csv', bins, ['length_group','recovered','total','rate'])
    save_table(output/'adjacent_links.csv', links, ['sequence','gt_track_id','previous_image','current_image','predictions_available','ambiguous','both_reliably_matched','recovered','previous_pred_id','current_pred_id'])
    save_table(output/'prediction_identity_mapping.csv', mixing, ['sequence','pred_track_id','gt_ids','gt_count'])
    make_plots(output, metrics, bins, tracks)
    return dict(schema_version=2, metrics={m['metric']: m for m in metrics}, counts=counts,
                scope='Recovery at GT annotated observations only; single-observation and conflicting GT tracks excluded. Ambiguous tracks remain in conservative denominator. Mixing checked only against reliable valid annotated GT identities.')


def make_plots(output, metrics, bins, tracks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter
    folder = output/'statistics'
    folder.mkdir()
    colors = ['#A9CCE3', '#A9D9C5', '#C8BCE4', '#F0C8AC', '#E5B8C8']
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'axes.edgecolor':'#CCD3DC',
                         'text.color':'#334155', 'axes.labelcolor':'#334155',
                         'xtick.color':'#475569','ytick.color':'#475569', 'figure.facecolor':'#FAFBFD'}):
        def save(fig, name):
            fig.savefig(folder/f'{name}.png', dpi=180, bbox_inches='tight')
            fig.savefig(folder/f'{name}.svg', bbox_inches='tight')
            plt.close(fig)
        lookup = {m['metric']:m for m in metrics}
        names = ['position_coverage','complete_gt_track_recovery','conditional_identity_retention','gt_link_recovery_conservative','gt_link_recovery_decidable']
        labels = ['Position coverage','Complete GT track success rate','Conditional identity retention','GT adjacent-link success (including ambiguity)','GT adjacent-link success (excluding ambiguity)']
        fig, ax = plt.subplots(figsize=(11,5), layout='constrained')
        for i, name in enumerate(names):
            m = lookup[name]; v = m['value']
            ax.barh(i, v or 0, color=colors[i], height=.62)
            ax.text(min((v or 0)+.015, .90), i, 'N/A (n=0)' if v is None else f"{v:.1%}  ({m['numerator']}/{m['denominator']})", va='center', fontsize=10)
        ax.set(yticks=range(5), yticklabels=labels, xlim=(0,1.18), title='Tracking evaluation | distinct denominators')
        ax.set_xticks([0,.25,.5,.75,1]); ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.invert_yaxis()
        ax.set_axisbelow(True); ax.grid(axis='x', alpha=.15)
        save(fig,'metric_overview')
        fig, axes = plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
        for i,b in enumerate(bins):
            axes[0].bar(i,b['rate'] or 0,color=colors[i],width=.6)
            axes[0].text(i,(b['rate'] or 0)+.025, f"{b['recovered']}/{b['total']}" if b['total'] else 'N/A', ha='center')
        axes[0].set(xticks=range(3),xticklabels=[b['length_group'] for b in bins],ylim=(0,1.12),title='Complete track success by GT length',xlabel='Number of annotated observations')
        axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        statuses=['recovered','not_recovered','needs_review','missing_prediction']
        values=[sum(t.get('recovery_status')==s for t in tracks) for s in statuses]
        axes[1].bar(range(4),values,color=colors[:4])
        axes[1].set(xticks=range(4),xticklabels=['Successful','Unsuccessful','Review','Missing files'],title='Eligible GT track outcomes',ylabel='Track count')
        for i,v in enumerate(values): axes[1].text(i,v,str(v),ha='center',va='bottom')
        axes[1].margins(y=.2)
        save(fig,'track_recovery')
        eligible=[t for t in tracks if t.get('recovery_eligible')]
        fig, axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        axes[0].hist([t['coverage'] for t in eligible],bins=np.linspace(0,1,11),color=colors[0],edgecolor='white')
        axes[0].set(xlim=(0,1),title='Position coverage per GT track',xlabel='Coverage',ylabel='Track count')
        axes[0].xaxis.set_major_formatter(PercentFormatter(1))
        ids=[t['distinct_pred_ids'] for t in eligible]
        axes[1].hist(ids,bins=np.arange(-.5,max(ids,default=0)+1.5),color=colors[2],edgecolor='white')
        axes[1].set(title='Prediction IDs per GT track',xlabel='Distinct prediction IDs',ylabel='Track count')
        save(fig,'track_distributions')
