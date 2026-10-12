"""Recompute every exported LUMEN rate/contrast and BFCL endpoint on CPU."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import gzip
import json
import numpy as np
from family_statistics import paired, rate

ROOT = Path(__file__).resolve().parents[1]

def check(actual, expected, fields):
    for field in fields:
        if field not in expected:
            continue
        if expected[field] is None:
            assert actual[field] is None, field
        else:
            assert np.allclose(actual[field], expected[field], atol=1e-9, rtol=0), (field, actual[field], expected[field])

def main():
    published=json.loads((ROOT/'data/current_final_results.json').read_text())
    contracts=json.loads((ROOT/'data/primary_contracts.json').read_text())
    total_rates=total_contrasts=total_capability=0
    for key,result in published.items():
        with gzip.open(ROOT/'data/response_metrics'/(key+'.json.gz'),'rt') as f:
            raw=json.load(f)
        aliases={r['id']:r['alias_of'] for r in contracts[key]['zero_dose_aliases']}
        groups=defaultdict(dict)
        for r in raw['observations']:
            component,context,condition,*ident=r['identity']
            assert tuple(ident) not in groups[component,context,condition]
            groups[component,context,condition][tuple(ident)]=r
        @lru_cache(maxsize=None)
        def sample(component,context,condition,language):
            actual=aliases.get(condition,condition)
            if component=='languages' and (component,context,actual) not in groups:
                component='main'
            return {i:r for i,r in groups[component,context,actual].items() if language=='all' or i[1]==language}
        for target in result['rates']:
            rows=sample(target['component'],target['context'],target['condition'],target['target_language'])
            ids=sorted(rows)
            got=rate([rows[i][target['metric']] for i in ids],[rows[i]['family'] for i in ids])
            check(got,target,['positive','resolved','missing','total','rate_percent','ci95_percent','all_response_bounds_percent'])
            total_rates+=1
        for target in result['contrasts']:
            a=sample(target['component'],target['context'],target['left'],target['target_language'])
            b=sample(target['component'],target['context'],target['right'],target['target_language'])
            assert set(a)==set(b)
            ids=sorted(a)
            same=aliases.get(target['left'],target['left'])==aliases.get(target['right'],target['right'])
            got=paired([a[i][target['metric']] for i in ids],[b[i][target['metric']] for i in ids],
                [a[i]['family'] for i in ids],multiplicity=target['multiplicity'],same_observations=same)
            check(got,target,['delta_pp','paired_total','paired_resolved','paired_missing','ci95_pp','bonferroni_ci_pp','all_response_delta_bounds_pp'])
            total_contrasts+=1
        controls={tuple(r['identity']):r for r in raw['controls']}
        for target in result['benign']:
            actual=aliases.get(target['condition'],target['condition'])
            ids=sorted(bid for (cid,bid),r in controls.items() if cid==actual and (target['stratum']=='all' or r['stratum']==target['stratum']))
            if not ids:
                assert target.get('total',0)==0
                continue
            metric=target['metric'];families=[controls[actual,i]['family'] for i in ids]
            values=[controls[actual,i][metric] for i in ids]
            got=rate(values,families)
            check(got,target,['positive','resolved','missing','rate_percent','ci95_percent'])
            change=paired(values,[controls['baseline',i][metric] for i in ids],families,same_observations=actual=='baseline')
            check(change,target,['delta_pp'])
            total_capability+=1
        print(key,'all endpoints reproduced',flush=True)
    print(json.dumps(dict(status='PASS',rates=total_rates,contrasts=total_contrasts,capability=total_capability)))

if __name__=='__main__':main()
