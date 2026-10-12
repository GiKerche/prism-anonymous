#!/usr/bin/env python3
"""Recompute all 24 primary differences from supplied response-level endpoints.

Requires Python 3 and NumPy. Run: python analysis/reproduce_primary.py
This checks sampling uncertainty conditional on the supplied judge labels;
it does not establish correctness of those labels.
"""
from pathlib import Path
import gzip,json
import numpy as np
from family_statistics import paired,rate

ROOT=Path(__file__).resolve().parents[1]
published=json.loads((ROOT/'data/current_final_results.json').read_text())
contracts=json.loads((ROOT/'data/primary_contracts.json').read_text())
verified=0
for key,result in published.items():
    with gzip.open(ROOT/'data/response_metrics'/(key+'.json.gz'),'rt') as f:raw=json.load(f)
    aliases={r['id']:r['alias_of'] for r in contracts[key]['zero_dose_aliases']}
    def sample(context,condition):
        actual=aliases.get(condition,condition)
        return {tuple(r['identity'][3:]):r for r in raw['observations'] if r['identity'][:3]==['main',context,actual]}
    for target in result['contrasts']:
        if (target['component'],target['target_language'],target['metric'])!=('main','all','semantic_unsafe'):continue
        a=sample(target['context'],target['left']);b=sample(target['context'],target['right'])
        keys=sorted(a);assert set(a)==set(b) and len(keys)==800
        same=aliases.get(target['left'],target['left'])==aliases.get(target['right'],target['right'])
        got=paired([a[k]['semantic_unsafe'] for k in keys],[b[k]['semantic_unsafe'] for k in keys],
            [a[k]['family'] for k in keys],multiplicity=target['multiplicity'],same_observations=same)
        for field in ['delta_pp','ci95_pp','bonferroni_ci_pp','paired_total','paired_resolved','paired_missing','families','all_response_delta_bounds_pp']:
            if target[field] is None:assert got[field] is None
            else:assert np.allclose(got[field],target[field],rtol=0,atol=1e-10),(key,field,got[field],target[field])
        verified+=1
    # All pooled physical-arm BFCL validity rates and paired changes.
    controls={tuple(r['identity']):r for r in raw['controls']}
    for target in result['benign']:
        if target['metric']!='official_valid':continue
        actual=aliases.get(target['condition'],target['condition'])
        ids=sorted(bid for (cid,bid),r in controls.items() if cid==actual and r['stratum']==target['stratum'])
        if not ids:continue
        families=[controls[actual,bid]['family'] for bid in ids]
        values=[controls[actual,bid]['official_valid'] for bid in ids]
        baseline=[controls['baseline',bid]['official_valid'] for bid in ids]
        got=rate(values,families);delta=paired(values,baseline,families,same_observations=actual=='baseline')
        assert got['positive']==target['positive'] and got['resolved']==target['resolved']
        assert np.isclose(delta['delta_pp'],target['delta_pp'],rtol=0,atol=1e-10)
    print(key,'primary contrasts and BFCL validity reproduced')
assert verified==24,verified
print('PASS:',verified,'primary paired contrasts reproduce, including missingness and adjusted intervals.')
