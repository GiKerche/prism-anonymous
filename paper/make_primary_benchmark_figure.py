#!/usr/bin/env python3
"""Render PRISM benchmark effects from frozen, selected-dose estimates.

Run python make_primary_benchmark_figure.py from any directory. No inference or fitting.
The primary image uses existing pointwise 95% CIs. A companion retains the
original multiplicity adjustments (BFCL stays 95%).
"""
from pathlib import Path
import csv
import math
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
HERE = ROOT / 'figures'
MODEL_Y = {
    'source_m1': .9, 'source_m3': 3.3, 'source_m5': 5.7,
    'receiver_m1': 9., 'receiver_m3': 11.4, 'receiver_m5': 13.8,
}
STYLES = {
    'native': {'color': '#24648B', 'marker': 'o', 'offset': -.22},
    'shadow': {'color': '#BD5D25', 'marker': 's', 'offset': .22},
    'transported': {'color': '#307D67', 'marker': 'D', 'offset': .22},
}
PANELS = [
    {'key': 'lumen', 'left': .241, 'width': .185,
     'title': '(a) LUMEN', 'subtitle': 'Unsafe-call proposals',
     'limits': (-55, 105), 'ticks': [-50, 0, 50, 100],
     'axis_label': 'More harmful calls →'},
    {'key': 'bfcl', 'left': .510, 'width': .155,
     'title': '(b) BFCL', 'subtitle': 'Tool-required accuracy',
     'limits': (-6, 18), 'ticks': [-5, 0, 5, 10, 15],
     'axis_label': 'Better tool use →'},
    {'key': 'soda', 'left': .754, 'width': .175,
     'title': '(c) SODA', 'subtitle': 'Unsafe executed trajectories',
     'limits': (-36, 52), 'ticks': [-25, 0, 25, 50],
     'axis_label': 'More harmful actions →'},
]


def signed(value):
    return f'{value:+.2f}'.replace('-', '−')


def tick_label(value, _position):
    return '0' if value == 0 else f'{value:+.0f}'.replace('-', '−')


def interval(record, metric, adjusted):
    if not adjusted or metric == 'bfcl':
        return record[metric]['ci95_pp']
    return record[metric]['simultaneous_ci_pp' if metric == 'soda'
                          else 'bonferroni_ci_pp']


def is_measured(record, metric):
    return (record['status'] == 'measured'
            and record[metric].get('status', 'measured') == 'measured')


def render(data, adjusted=False):
    # Use all measured BFCL intervals to set a shared, unclipped scale.
    bounds = [edge for r in data['rows'] if is_measured(r, 'bfcl') for edge in r['bfcl']['ci95_pp']]
    panel = next(p for p in PANELS if p['key'] == 'bfcl')
    lower = math.floor(min(-1, min(bounds)-1)/2)*2
    upper = math.ceil(max(1, max(bounds)+1)/2)*2
    step = 2 if upper-lower <= 16 else 5
    panel['limits'] = (lower, upper)
    panel['ticks'] = list(range(math.ceil(lower/step)*step, upper+1, step))
    fig = plt.figure(figsize=(6.75, 4.50), facecolor='white')
    body_bottom, body_height = .123, .723
    axes = []
    for panel in PANELS:
        axis = fig.add_axes([panel['left'], body_bottom, panel['width'], body_height])
        axes.append(axis)
        axis.set_ylim(15., -.5)
        axis.set_xlim(*panel['limits'])
        axis.set_xticks(panel['ticks'])
        axis.xaxis.set_major_formatter(FuncFormatter(tick_label))
        axis.set_yticks([])
        axis.tick_params(axis='x', length=2, width=.5, color='#9CA3AA')
        axis.set_axisbelow(True)
        axis.grid(axis='x', color='#E4E8EC', linewidth=.5)
        axis.axvline(0, color='#87929C', linewidth=.7, zorder=1)
        for side in ('left', 'right', 'top'):
            axis.spines[side].set_visible(False)
        axis.spines['bottom'].set_color('#B9C0C6')
        axis.spines['bottom'].set_linewidth(.5)
        axis.text(.5, 1.070, panel['title'], transform=axis.transAxes,
                  ha='center', fontsize=8.1, fontweight='bold')
        axis.text(.5, 1.028, panel['subtitle'], transform=axis.transAxes,
                  ha='center', fontsize=6.8)
        axis.set_xlabel(panel['axis_label'], fontsize=6.6, labelpad=3)
        for key, y in MODEL_Y.items():
            if key in ('source_m1', 'source_m5', 'receiver_m3'):
                axis.axhspan(y - 1.06, y + 1.06, color='#F5F7F9', zorder=0)
        for record in data['rows']:
            metric = panel['key']
            if not is_measured(record, metric):
                continue
            style = STYLES[record['estimator']]
            y = (MODEL_Y[record['model_key']]
                 + (-.55 if record['sign'] == 'plus' else .55) + style['offset'])
            value = record[metric]['delta_pp']
            lower, upper = interval(record, metric, adjusted)
            assert panel['limits'][0] < lower <= value <= upper < panel['limits'][1]
            axis.errorbar(value, y, xerr=[[value-lower], [upper-value]],
                          fmt=style['marker'], color=style['color'],
                          markersize=3.3, linewidth=.85,
                          capsize=1.6, capthick=.75, zorder=3)
            axis.text(1.033, y, signed(value), transform=axis.get_yaxis_transform(),
                      ha='left', va='center', color=style['color'],
                      fontsize=6.35, clip_on=False)
        for key in ('source_m1', 'receiver_m1'):
            if panel['key'] == 'soda' and key.startswith('receiver'):
                continue
            axis.text(.5, MODEL_Y[key] + .55, 'No suppression dose',
                      transform=axis.get_yaxis_transform(), ha='center', va='center',
                      fontsize=6.15, color='#5E6770',
                      bbox={'facecolor': '#F5F7F9' if key == 'source_m1' else 'white',
                            'edgecolor': 'none', 'pad': .5})

    # Names align all benchmarks. BFCL baselines contextualize its changes.
    left_axis = axes[0]
    label_x = (.013 - PANELS[0]['left']) / PANELS[0]['width']
    sign_x = (.215 - PANELS[0]['left']) / PANELS[0]['width']
    for key, y in MODEL_Y.items():
        left_axis.text(label_x, y - .14, data['models'][key],
                       transform=left_axis.get_yaxis_transform(), ha='left', va='center',
                       fontsize=7.4, fontweight='bold', color='#263440', clip_on=False)
        base = data['bfcl_baselines'][key]
        left_axis.text(label_x, y + .32, f'BFCL baseline: {base:.2f}%',
                       transform=left_axis.get_yaxis_transform(), ha='left', va='center',
                       fontsize=6., color='#596775', clip_on=False)
        for offset, sign in [(-.55, '+'), (.55, '−')]:
            left_axis.text(sign_x, y + offset, sign,
                           transform=left_axis.get_yaxis_transform(), ha='center', va='center',
                           fontsize=9, color='#455363', clip_on=False)

    # Missing receiver SODA data are not zero-effect measurements.
    soda_axis = axes[2]
    soda_axis.axhspan(7.95, 14.9, color='white', zorder=4)
    soda_axis.text(.5, 11.25, 'Not evaluated\nin larger models',
                   transform=soda_axis.get_yaxis_transform(), ha='center', va='center',
                   fontsize=7.5, color='#65717D', linespacing=1.6, zorder=5)
    handles = [Line2D([], [], color=STYLES[branch]['color'],
                      marker=STYLES[branch]['marker'], linestyle='none',
                      markersize=3.8, label=label)
               for branch, label in [
                   ('native', 'Native (BF16 labels)'),
                   ('shadow', 'Shadow (NF4 labels)'),
                   ('transported', 'Transported from smaller model')]]
    fig.legend(handles=handles, loc='upper center', ncol=3,
               bbox_to_anchor=(.51, 1.001), frameon=False,
               handletextpad=.4, columnspacing=1.3, fontsize=7)

    def page_y(y):
        return body_bottom + body_height * (15. - y) / 15.5

    fig.text(.013, .881, 'Smaller models', fontsize=7.4,
             fontweight='bold', color='#455363')
    fig.text(.215, .866, 'Dose', ha='center', fontsize=6., color='#455363')
    separator_y = page_y(7.42)
    fig.add_artist(Line2D([.013, .994], [separator_y, separator_y],
                         transform=fig.transFigure, color='#BEC7CF', lw=.65))
    fig.text(.013, page_y(7.76), 'Larger models', ha='left', va='center',
             fontsize=7.4, fontweight='bold', color='#455363')
    ci_label = ('Original adjusted intervals; BFCL: 95% CIs' if adjusted
                else 'Pointwise 95% CIs')
    fig.text(.5, .035, f'Change from baseline (percentage points)  |  {ci_label}',
             ha='center', va='center', fontsize=7.1)
    stem = 'reviewer_primary_effects' + ('_adjusted' if adjusted else '')
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(HERE / f'{stem}.{ext}', dpi=260, facecolor='white', pad_inches=0)
    plt.close(fig)


def write_csv(data):
    fields = ['model', 'estimator', 'sign', 'alpha', 'benchmark', 'status',
              'delta_pp', 'ci95_lower_pp', 'ci95_upper_pp',
              'adjusted_lower_pp', 'adjusted_upper_pp', 'adjustment_family_size',
              'bfcl_baseline_percent', 'bfcl_accuracy_percent']
    with (HERE / 'plotted_effects.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in data['rows']:
            for metric in ('lumen', 'bfcl', 'soda'):
                row = {key: record[key] for key in ('model', 'estimator', 'sign', 'alpha')}
                row.update(benchmark=metric, status=record[metric].get('status', record['status']))
                if is_measured(record, metric):
                    lo, hi = interval(record, metric, False)
                    alo, ahi = interval(record, metric, True)
                    row.update(delta_pp=record[metric]['delta_pp'],
                               ci95_lower_pp=lo, ci95_upper_pp=hi,
                               adjusted_lower_pp=alo, adjusted_upper_pp=ahi,
                               adjustment_family_size=record[metric].get(
                                   'multiplicity', record[metric].get('multiplicity_family_size')))
                    if metric == 'bfcl':
                        row.update(bfcl_baseline_percent=data['bfcl_baselines'][record['model_key']],
                                   bfcl_accuracy_percent=record[metric]['accuracy_percent'])
                writer.writerow(row)


def main():
    data = json.loads((ROOT / 'data/benchmark_effects.json').read_text())
    for panel in PANELS:
        values = [bound for row in data['rows'] if is_measured(row, panel['key']) for adjusted in [False, True] for bound in interval(row, panel['key'], adjusted)]
        if min(values) <= panel['limits'][0] or max(values) >= panel['limits'][1]:
            from math import floor, ceil
            step = 10 if panel['key'] == 'bfcl' else 25
            lo, hi = step * floor((min(values)-1)/step), step * ceil((max(values)+1)/step)
            panel['limits'] = (min(lo, 0), max(hi, 0))
            panel['ticks'] = list(range(int(panel['limits'][0]), int(panel['limits'][1])+1, step))
    assert len(data['rows']) == 24
    assert sum(r['status'] == 'no_selected_dose' for r in data['rows']) == 4
    assert [sum(is_measured(r, m) for r in data['rows'])
            for m in ('lumen', 'bfcl', 'soda')] == [20, 20, 10]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                        'xtick.labelsize': 6.4, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                        'svg.fonttype': 'none', 'axes.unicode_minus': True})
    render(data, adjusted=False)
    render(data, adjusted=True)
    write_csv(data)
    print('Rendered 50 measured effects across LUMEN, BFCL and SODA; '
          'pointwise and original-adjustment versions.')


if __name__ == '__main__':
    main()
