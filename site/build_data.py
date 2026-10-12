"""Build the browser's small numerical catalog from deposited scientific exports.

Recorded response shards are distributed separately in docs/data/responses.
No model, evaluator, GPU, network service, or private input is needed here.
"""
from pathlib import Path
import csv
import json
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'paper/data'
OUT = ROOT / 'docs/data'
OUT.mkdir(parents=True, exist_ok=True)

def read(name):
    return json.loads((DATA / name).read_text())

def keep(row, names):
    return {n: row[n] for n in names if n in row}

rate_fields = ['component','context','target_language','condition','actual_condition',
               'positive','resolved','total','missing','rate_percent','coverage_percent',
               'all_response_bounds_percent','ci95_percent']
effect_fields = ['component','context','target_language','left','right','condition',
                 'delta_pp','paired_resolved','paired_total','paired_missing','ci95_pp',
                 'bonferroni_ci_pp','multiplicity','all_response_delta_bounds_pp',
                 'exact_baseline_alias']
protocol = json.loads((ROOT / 'reproduction/protocol.json').read_text())
names = dict(zip(protocol['models'], ['Qwen3.5-9B','Ministral-3-8B','Llama-3.1-8B',
                                     'Qwen3.5-27B','Ministral-3-14B','Llama-3.1-70B']))
# Use explicit keys so JSON dictionary order cannot change model names.
names = {'source_m1':'Qwen3.5-9B','source_m3':'Ministral-3-8B','source_m5':'Llama-3.1-8B',
         'receiver_m1':'Qwen3.5-27B','receiver_m3':'Ministral-3-14B','receiver_m5':'Llama-3.1-70B'}
catalog = {'models':[], 'results':{}, 'controls':{}, 'doses':{}, 'projections':{}}
for key, m in protocol['models'].items():
    catalog['models'].append(dict(key=key, name=names[key], **keep(m,
        ['layer','blocks','sigma','selected_alphas','model_id','model_revision'])))

for key, result in read('current_final_results.json').items():
    catalog['results'][key] = {
        'rates':[keep(r,rate_fields) for r in result['rates'] if r['metric']=='semantic_unsafe'],
        'contrasts':[keep(r,effect_fields) for r in result['contrasts'] if r['metric']=='semantic_unsafe'],
        'benign':[keep(r,rate_fields+effect_fields) for r in result['benign']
                  if r.get('stratum')=='tool_required' and r.get('metric')=='official_valid']}

controls = read('current_receiver_controls.json')
for key in names:
    if not key.startswith('receiver'): continue
    catalog['controls'][key] = {
        'arms':[dict(**keep(r,rate_fields+['alpha']), capability=r['capability'])
                for r in controls['arms'] if r['key']==key and
                r['condition'].startswith('selected_receiver_native')],
        'contrasts':[keep(r,effect_fields) for r in controls['contrasts']
                     if r['key']==key and r['condition'].startswith('selected_receiver_native')
                     and r.get('reference')=='baseline']}

for key, grid in read('development_curves.json').items():
    rows=[]
    for r in grid['curve']:
        b=grid['benign'].get(r['condition'],{}).get('tool_required',{}).get('official_valid',{})
        rows.append(dict(**keep(r,['condition','context','source','alpha','admissible',
                                   'rejection_reasons','signed_gain_pp','call_proxy']), capability=b))
    catalog['doses'][key] = {'rows':rows, 'selected':grid['selected_alphas']}

catalog['geometry'] = read('language_geometry.json')['rows']
catalog['language_summary'] = read('language_effects.json')['rows']
with (DATA/'language_cells.csv').open() as f:
    catalog['language_cells'] = [{k:(v if k in ['model','estimator','source_language',
        'target_language','context'] else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]

for key in names:
    if not key.startswith('source'): continue
    with np.load(ROOT/'reproduction'/key/'directions.npz',allow_pickle=False) as z:
        labels=[s+'_'+l for s in ['native','shadow'] for l in ['pooled','en','pt','es','zh','hi']]
        X=np.stack([z[n] for n in labels]).astype(float)
    X-=X.mean(axis=0)
    u,s,_=np.linalg.svd(X,full_matrices=False)
    xy=u[:,:2]*s[:2]
    # Deterministic display orientation; distances in this projection are descriptive.
    for j in range(2):
        if xy[np.argmax(abs(xy[:,j])),j]<0: xy[:,j]*=-1
    catalog['projections'][key] = {
        'points':dict(zip(labels,xy.tolist())),
        'variance_explained':float((s[:2]**2).sum()/(s**2).sum()),
        'description':'Two principal components of centered unit direction vectors; display only.'}

example=json.loads((ROOT/'paper/task/observed_intervention_example.json').read_text())
catalog['example']={k:example[k] for k in ['scenario_id','language','seed','context','user_prompt'] if k in example}
(OUT/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n')

cases=json.loads((ROOT/'paper/task/scenario_inputs.json').read_text())['cases']
cases=[keep(r,['scenario_id','language','category','family','context','user_prompt','system_prompt'])
       for r in cases if r['partition']=='final_test']
(OUT/'prompts.json').write_text(json.dumps(cases,ensure_ascii=False,separators=(',',':'))+'\n')
shutil.copyfile(ROOT/'paper/task/tools_by_language.json',OUT/'tools.json')
(ROOT/'docs/assets').mkdir(exist_ok=True)
shutil.copyfile(ROOT/'paper/main.pdf',ROOT/'docs/assets/paper.pdf')
print(f'Built catalog for {len(names)} models, {len(cases)} prompts, and {len(catalog["language_cells"])} language cells.')
