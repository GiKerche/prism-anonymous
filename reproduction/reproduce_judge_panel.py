"""Recompute the complete Llama70 evaluator sensitivity from stored votes."""
from collections import defaultdict
from pathlib import Path
import gzip
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'paper/analysis'))
from family_statistics import rate,paired

def main():
    with gzip.open(ROOT/'llama70_judge_panel.json.gz','rt') as f:rows=json.load(f)
    result=json.loads((ROOT.parent/'paper/data/llama70_full_judges.json').read_text())
    groups=defaultdict(dict)
    for r in rows:groups[r['context'],r['condition']][r['scenario_id'],r['language'],r['seed']]=r
    for target in result['rates']:
        sample=groups[target['context'],target['condition']];ids=sorted(sample)
        got=rate([sample[i][target['judge']] for i in ids],[sample[i]['family'] for i in ids])
        for field in ['positive','resolved','missing','rate_percent','ci95_percent']:
            assert np.allclose(got[field],target[field],atol=1e-9,rtol=0),(field,got[field],target[field])
    for target in result['contrasts']:
        a=groups[target['context'],target['condition']];b=groups[target['context'],'baseline'];ids=sorted(a)
        assert set(a)==set(b)
        got=paired([a[i][target['judge']] for i in ids],[b[i][target['judge']] for i in ids],[a[i]['family'] for i in ids],multiplicity=8)
        for field in ['delta_pp','ci95_pp','bonferroni_ci_pp','all_response_delta_bounds_pp']:
            assert np.allclose(got[field],target[field],atol=1e-9,rtol=0),(field,got[field],target[field])
    candidates=json.loads((ROOT/'llama70_candidate_votes.json').read_text())
    assert len(candidates)==952
    for target in result['disagreement']:
        sample=candidates if target['context']=='all' else [r for r in candidates if (r['context'],r['condition'])==(target['context'],target['condition'])]
        valid=[r for r in sample if r[target['left']] is not None and r[target['right']] is not None]
        assert len(sample)==target['candidates'] and len(valid)==target['jointly_resolved']
        assert sum(r[target['left']]!=r[target['right']] for r in valid)==target['disagree']
    print(json.dumps(dict(status='PASS',responses=len(rows),candidates=len(candidates),rates=len(result['rates']),contrasts=len(result['contrasts']))))

if __name__=='__main__':main()
