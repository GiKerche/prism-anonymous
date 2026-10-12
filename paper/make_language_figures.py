"""Render all current source-to-target language effects from frozen aggregates."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data/current_final_results.json').read_text())
LANGS = ['pt', 'en', 'es', 'zh', 'hi']
MODELS = [('source_m1', 'Qwen3.5-9B'), ('source_m3', 'Ministral-3-8B'),
          ('source_m5', 'Llama-3.1-8B')]
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans',
                     'pdf.fonttype': 42})

for sign, context in [('plus', 'ethical'), ('minus', 'neutral')]:
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), constrained_layout=True)
    for col, (key, model) in enumerate(MODELS):
        for row, estimator in enumerate(['native', 'shadow']):
            ax = axes[row, col]
            values = np.zeros((5, 5))
            for i, source in enumerate(LANGS):
                for j, target in enumerate(LANGS):
                    matches = [x for x in DATA[key]['contrasts']
                               if x['component'] == 'languages'
                               and x['context'] == context
                               and x['target_language'] == target
                               and x['metric'] == 'semantic_unsafe'
                               and x['left'] == f'{estimator}_{source}_{sign}'
                               and x['right'] == 'baseline']
                    assert len(matches) == 1
                    effect = matches[0]
                    assert effect['paired_total'] == 160
                    values[i, j] = effect['delta_pp']
                    if key == 'source_m1' and sign == 'minus':
                        assert effect['exact_baseline_alias'] and effect['delta_pp'] == 0
            im = ax.imshow(values, cmap='RdBu_r', norm=TwoSlopeNorm(0, -90, 90))
            for i in range(5):
                for j in range(5):
                    v = values[i, j]
                    ax.text(j, i, f'{v:+.1f}' if v else '0.0', ha='center', va='center',
                            fontsize=9.5, color='white' if abs(v) >= 55 else '#111111')
            alpha = DATA[key]['selected_alphas'][f'{estimator}_{sign}']
            title = f'{model} | {estimator}\n'
            title += 'a = 0; baseline alias' if alpha == 0 else f'a = {alpha:+g}'
            ax.set_title(title, fontsize=10)
            ax.set_xticks(range(5), LANGS)
            ax.set_yticks(range(5), LANGS)
            ax.set_xlabel('Target language')
            if col == 0:
                ax.set_ylabel('Vector language')
            ax.tick_params(length=0)
            ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
            ax.grid(which='minor', color='white', linewidth=.6)
            ax.tick_params(which='minor', bottom=False, left=False)
    bar = fig.colorbar(im, ax=axes, shrink=.85, pad=.02, ticks=[-90, -60, -30, 0, 30, 60, 90])
    bar.set_label('Paired unsafe-action change (percentage points)')
    fig.savefig(ROOT / f'figures/current_language_{sign}.pdf',
                metadata={'Title': f'Current language transfer: {sign}',
                          'Author': 'Anonymous authors', 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
