"""Reconstruct frozen directions, predictions, layers and maps from numeric data.

No model downloads, GPU work or judge calls. --refit-probes additionally
refits every construction-only logistic regression using the pinned recipe.
"""
from collections import Counter
from pathlib import Path
import argparse
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
LANGS=['pt','en','es','zh','hi']

def read(p):return json.loads(p.read_text())
def load(p,name='activations'):
    with np.load(p,allow_pickle=False) as a:return a[name]

def observations(rows,index):
    rows=[r for r in rows if r['label'] is not None]
    prompts=Counter((r['scenario_id'],r['language']) for r in rows)
    scenarios=Counter(s for s,l in prompts)
    weights=np.array([5./scenarios[r['scenario_id']]/prompts[r['scenario_id'],r['language']] for r in rows])
    ids=np.array([index[r['scenario_id'],r['language']] for r in rows])
    return ids,np.array([r['label'] for r in rows]),weights

def fractions(rows,scenarios):
    lookup={(r['scenario_id'],r['language'],r['seed']):r['label'] for r in rows}
    return np.array([[np.mean(v) if all(y is not None for y in v) else np.nan
        for lang in LANGS for v in [[lookup[s,lang,seed] for seed in [101,102,103,104]]]] for s in scenarios])

def contribution(h,z,indices,j):
    indices=indices[np.isfinite(z[indices,j])]
    if len(indices)<2:return None
    values=z[indices,j].astype(np.float64); centered=values-values.mean()
    if not np.any(centered!=0):return None
    value=(centered@h[indices,j].astype(np.float64))/len(indices)
    norm=np.linalg.norm(value)
    return value if np.isfinite(norm) and norm>np.finfo(float).tiny else None

def fit_direction(h,z,indices=None):
    if indices is None:indices=np.arange(len(z))
    values=[contribution(h,z,indices,j) for j in range(5)]
    values=[v for v in values if v is not None]
    if not values:raise ValueError('Direction unavailable')
    v=np.sum(values,axis=0);norm=np.linalg.norm(v)
    if not np.isfinite(norm) or norm<=np.finfo(float).tiny:raise ValueError('Direction unavailable')
    return (v/norm).astype(np.float32)

def order_index(rows):return {(r['scenario_id'],r['language']):r['index'] for r in rows}

def main(refit=False):
    protocol=read(ROOT/'protocol.json');checked=0
    for key,spec in protocol['models'].items():
        base=ROOT/key;fits=np.load(base/'frozen_fits.npz',allow_pickle=False)
        labels=read(base/'labels.json')
        order=read(base/'construction_activations/order.json');ix=order_index(order)
        vidx=order_index(read(base/'validation_order.json'))
        ids,y,w=observations(labels['validation_i']['bf16'],vidx)
        scores=load(base/'validation_scores.npz','scores');curve=[]
        buildids,by,bw=observations(labels['construction']['bf16'],ix)
        mass=np.bincount(by,weights=bw,minlength=2);cw=bw.sum()/(2*mass)
        pw=np.bincount(buildids,weights=bw,minlength=len(order));valid=pw>0
        for layer in range(spec['blocks']):
            pre=f'layer_{layer:02d}_';x=load(base/'construction_activations'/f'layer_{layer:02d}.npz')
            if refit:
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                scaler=StandardScaler().fit(x[valid],sample_weight=pw[valid])
                model=LogisticRegression(**{k:v for k,v in protocol['classifier'].items() if k not in ['type','threshold']})
                model.fit(scaler.transform(x[buildids]),by,sample_weight=bw*cw[by])
                for name,value in [('mean',scaler.mean_),('scale',scaler.scale_),('coef',model.coef_[0]),('intercept',model.intercept_)]:
                    assert np.allclose(value,fits[pre+name],rtol=1e-6,atol=1e-7),(key,layer,name)
            xv=load(base/'validation_activations'/f'layer_{layer:02d}.npz')
            validation_order=order_index(read(base/'validation_activations/order.json'))
            permutation=[validation_order[s,l] for (s,l),i in sorted(vidx.items(),key=lambda kv:kv[1])]
            logits=((xv[permutation]-fits[pre+'mean'])/fits[pre+'scale'])@fits[pre+'coef']+fits[pre+'intercept'][0]
            predicted=1/(1+np.exp(-np.clip(logits,-700,700)))
            assert np.allclose(predicted,scores[layer],rtol=1e-9,atol=1e-9),(key,layer,'prediction')
            pred=scores[layer,ids]>=.5
            ba=np.mean([w[(y==c)&(pred==c)].sum()/w[y==c].sum() for c in [0,1]])
            curve.append(float(ba));checked+=1
        selected=read(base/'layer_selection.json')
        assert int(np.argmax(curve))==spec['layer']==selected['selected_layer'],(key,'layer')
        assert np.isclose(curve[spec['layer']],selected['best_validation']['balanced_accuracy'],atol=1e-12)
        directions=np.load(base/'directions.npz',allow_pickle=False)
        assert all(abs(np.linalg.norm(directions[k].astype(float))-1)<2e-5 for k in directions.files)
        x=load(base/'construction_activations'/f'layer_{spec["layer"]:02d}.npz')
        scenarios=sorted({r['scenario_id'] for r in order})
        h=np.stack([[x[ix[s,l]] for l in LANGS] for s in scenarios])
        for precision,branch in [('bf16','native'),('nf4','shadow')]:
            if precision not in labels['construction']:continue
            z=fractions(labels['construction'][precision],scenarios)
            name=branch+'_pooled' if key.startswith('source') else 'receiver_native'
            d=fit_direction(h,z)
            assert np.allclose(d,directions[name],atol=2e-7,rtol=1e-6),(key,name)
            if key.startswith('source'):
                for j,lang in enumerate(LANGS):
                    v=contribution(h,z,np.arange(len(z)),j); v=(v/np.linalg.norm(v)).astype(np.float32)
                    assert np.allclose(v,directions[branch+'_'+lang],atol=2e-7,rtol=1e-6)
                if branch=='native':
                    projections=np.array([float(h[i,j]@d) for i,j in zip(*np.nonzero(np.isfinite(z)))])
                    sigma=1.4826*np.median(abs(projections-np.median(projections)))
                    assert np.isclose(sigma,spec['sigma'],atol=1e-6,rtol=1e-6),(key,'sigma',sigma,spec['sigma'])
        if key.startswith('receiver'):
            m=np.load(base/'map_arrays.npz',allow_pickle=False)
            source=load(ROOT/key.replace('receiver','source')/'directions.npz','native_pooled').astype(float)
            d=((source@m['source_basis'])@m['weights'])@m['receiver_basis'].T
            d=(d/np.linalg.norm(d)).astype('<f4');d/=np.linalg.norm(d.astype(float))
            assert np.allclose(d,directions['transported_native'],atol=1e-7)
            anchors=np.concatenate([load(p) for p in sorted((base/'anchors').glob('receiver_*.npz'))])[:1024].astype(float)
            projection=anchors@d.astype(float)
            sigma=1.4826*np.median(abs(projection-np.median(projection)))
            assert np.isclose(sigma,spec['sigma'],rtol=1e-8,atol=1e-8),(key,'sigma')
        print(key,'directions, predictions, selected layer and scale reproduced',flush=True)
    print(json.dumps(dict(status='PASS',layers=checked,construction_probes_refitted=refit)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--refit-probes',action='store_true')
    main(parser.parse_args().refit_probes)
