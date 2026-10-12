"""Render the frozen Validation I layer curves from the distributed metrics."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root/'data/probe_selection_all_layers.csv').open() as handle:
    rows = list(csv.DictReader(handle))
models = ['Qwen3.5-9B', 'Ministral-3-8B', 'Llama-3.1-8B']
colors = ['#2563a6', '#b86b19', '#287f70']
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans',
                     'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.7), sharey=True, constrained_layout=True)
for ax, name, color in zip(axes, models, colors):
    values = [r for r in rows if r['model'] == name]
    x = [int(r['layer']) for r in values]
    y = [100*float(r['balanced_accuracy']) for r in values]
    selected = next(r for r in values if r['selected'] == '1')
    layer = int(selected['layer']); score = 100*float(selected['balanced_accuracy'])
    ax.axhline(50, color='#999999', linestyle='--', linewidth=.8)
    ax.plot(x, y, color=color, linewidth=1.5)
    ax.scatter([layer], [score], s=70, marker='*', color=color, zorder=3)
    ax.set_title(f'{name} | layer {layer}', fontsize=10)
    ax.set_xlabel('Layer (zero-based)')
    ax.set_xlim(0, max(x)); ax.set_ylim(0, 102)
    ax.set_xticks([0, 8, 16, 24, max(x)])
    ax.grid(axis='y', color='#dddddd', linewidth=.5)
axes[0].set_ylabel('Balanced accuracy (%)')
output = root/'figures/probe_selection_final_cycle.pdf'
output.parent.mkdir(exist_ok=True)
fig.savefig(output, metadata={'Title': 'Full-depth probe selection in the final evaluation cycle',
                             'Author': 'Anonymous authors', 'CreationDate': None, 'ModDate': None})
plt.close(fig)
