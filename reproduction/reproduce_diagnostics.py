"""Reproduce audited label yield, scarcity and frozen language geometry."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from reproduce_selection import read,load,order_index,fractions,fit_direction,LANGS

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'paper/data'

def main():
    protocol=read(ROOT/'protocol.json')
    split=read(ROOT.parent/'paper/task/partitions.json')
    scenarios=sorted(split['roles']['build'])
    families=sorted({split['family_by_scenario'][s] for s in scenarios})
    family_indices=[[i for i,s in enumerate(scenarios) if split['family_by_scenario'][s]==f] for f in families]
    rng=np.random.default_rng(20261001)
    draws={n:[np.concatenate([family_indices[j] for j in rng.choice(len(families),n,replace=False)])
              for _ in range(1000)] for n in [2,4,8,16,24,len(families)]}
    changes=read(ROOT/'audited_label_changes.json')
    expected_yield=read(DATA/'label_yield.json');expected_scarcity=read(DATA/'scarce_labels.json')
    count=0
    for slot in ['m1','m3','m5']:
        key='source_'+slot;base=ROOT/key
        labels=deepcopy(read(base/'labels.json')['construction'])
        for c in changes:
            if c['key']!=key or c['partition']!='build':continue
            r=next(r for r in labels[c['precision']] if all(r[k]==c[k] for k in ['scenario_id','language','seed']))
            assert r['label']==c['old_label'];r['label']=c['corrected_label']
        for target in expected_yield['rows']:
            if target['slot']!=slot:continue
            rows=[r for r in labels[target['precision']] if target['language']=='all' or r['language']==target['language']]
            positive=[r for r in rows if r['label']==1];valid=[r for r in rows if r['label'] is not None]
            assert len(positive)==target['positive'] and len(valid)==target['resolved']
            assert len({r['scenario_id'] for r in positive})==target['positive_scenarios']
            assert len({split['family_by_scenario'][r['scenario_id']] for r in positive})==target['positive_families']
        layer=protocol['models'][key]['layer'];x=load(base/'construction_activations'/f'layer_{layer:02d}.npz')
        ix=order_index(read(base/'construction_activations/order.json'))
        h=np.stack([[x[ix[s,l]] for l in LANGS] for s in scenarios])
        for precision in ['bf16','nf4']:
            z=fractions(labels[precision],scenarios);full=fit_direction(h,z)
            for budget,samples in draws.items():
                undefined=0;no_positive=0;support=[];cosines=[]
                for ids in samples:
                    selected=z[ids];no_positive+=not bool(np.any(selected>0))
                    support.append(sum(np.isfinite(selected[:,j]).sum()>=2 and np.ptp(selected[np.isfinite(selected[:,j]),j])>0 for j in range(5)))
                    try:d=fit_direction(h,z,ids)
                    except ValueError:undefined+=1
                    else:cosines.append(float(d.astype(float)@full.astype(float)))
                target=next(r for r in expected_scarcity['rows'] if (r['slot'],r['precision'],r['families'])==(slot,precision,budget))
                assert undefined==target['undefined_fits'],(key,precision,budget)
                assert np.isclose(100*no_positive/1000,target['no_positive_percent'])
                assert np.isclose(np.mean(support),target['mean_identifiable_languages'])
                assert np.isclose(np.median(cosines),target['median_full_build_cosine'],atol=1e-9,rtol=0)
                count+=1
        vectors=np.load(base/'directions.npz',allow_pickle=False)
        for branch in ['native','shadow']:
            v=np.stack([vectors[branch+'_'+lang] for lang in LANGS]).astype(float)
            target=next(r for r in read(DATA/'language_geometry.json')['rows'] if (r['slot'],r['estimator'])==(slot,branch))
            assert np.allclose(v@v.T,target['cosine'],atol=1e-12,rtol=0)
        print(key,'audited yields, paired scarcity and frozen geometry reproduced',flush=True)
    print(json.dumps(dict(status='PASS',scarcity_rows=count,label_yield_rows=len(expected_yield['rows']))))

if __name__=='__main__':main()
