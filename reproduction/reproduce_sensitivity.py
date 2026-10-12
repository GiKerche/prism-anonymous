"""Verify the fixed audit, language summaries and exploratory equivalence check."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'paper/analysis'))
from family_statistics import interval


def read(path):return json.loads(path.read_text())


def check(actual,expected,fields):
    for field in fields:
        a,b=actual[field],expected[field]
        if a is None or b is None:assert a is b,(field,a,b)
        else:assert np.allclose(a,b,atol=1e-9,rtol=0),(field,a,b)


def audit():
    data=read(ROOT/'reproduction/stratified_audit.json')
    expected=read(ROOT/'paper/data/judge_sensitivity.json');rates={}
    for key,model in data.items():
        # Reconstruct agreement indicators from the three votes.
        for row in model['rows']:
            a,b,c=row['votes'];ys=[a,b,c]
            consensus=a if a is not None and b is not None and a==b else None
            majority=1 if ys.count(1)>=2 else 0 if ys.count(0)>=2 else None
            values=dict(a_b_disagreement=None if a is None or b is None else a!=b,
                        c_consensus_agreement=None if c is None or consensus is None else c==consensus,
                        c_panel_agreement=None if c is None or majority is None else c==majority)
            check(values,row,values)
        for target in model['summary']:
            rows=[r for r in model['rows'] if target['language']=='all' or r['language']==target['language']]
            metric=target['metric'];valid=[r for r in rows if r[metric] is not None]
            weight=sum(r['inverse_probability_weight'] for r in valid)
            count=sum(r[metric] for r in valid)
            got=dict(sampled=len(rows),valid=len(valid),missing=len(rows)-len(valid),count=count,
                raw_percent=100*count/len(valid) if valid else None,
                inverse_inclusion_weighted_percent=100*sum(r[metric]*r['inverse_probability_weight'] for r in valid)/weight if weight else None)
            check(got,target,got)
        for target in [r for r in expected['rates'] if r['key']==key]:
            arm=next(a for a in model['arms'] if a['context']==target['context'] and a['condition']==target['actual_condition'])
            selected=[model['rows'][i] for i in arm['selected_indices']]
            lower=upper=positive=resolved=0.;missing=observed_pos=observed_n=0
            for stratum in arm['strata']:
                sample=[r for r in selected if r['stratum']==stratum['stratum']]
                den=sum(r['inverse_probability_weight'] for r in sample);assert den>0
                yes=absent=0.
                for r in sample:
                    votes=r['votes']
                    y=(1 if votes.count(1)>=2 else 0 if votes.count(0)>=2 else None) if target['judge']=='majority_abc' else votes[['claude','grok','deepseek_audit'].index(target['judge'])]
                    weight=r['inverse_probability_weight']
                    if y is None:absent+=weight;missing+=1
                    else:yes+=weight*y;observed_pos+=y;observed_n+=1
                count=stratum['count'];lower+=count*yes/den;upper+=count*(yes+absent)/den
                positive+=count*yes/den;resolved+=count*(den-absent)/den
            denominator=arm['total']-arm['candidates']+resolved
            got=dict(rate_percent=100*positive/denominator if denominator else None,
                bounds_percent=[100*lower/arm['total'],100*upper/arm['total']],sampled_candidates=len(selected),
                valid_sample_votes=observed_n,positive_sample_votes=observed_pos,missing_votes=missing,
                population_candidates=arm['candidates'],population_responses=arm['total'])
            check(got,target,got);rates[key,target['context'],target['condition'],target['judge']]=got
    for target in expected['contrasts']:
        prefix=target['key'],target['context'];judge=target['judge']
        a=rates[prefix+(target['condition'],judge)];b=rates[prefix+('baseline',judge)]
        got=dict(delta_pp=a['rate_percent']-b['rate_percent'],
            missing_vote_bounds_pp=[a['bounds_percent'][0]-b['bounds_percent'][1],a['bounds_percent'][1]-b['bounds_percent'][0]])
        check(got,target,got)
    return dict(audit_responses=sum(len(m['rows']) for m in data.values()),audit_rates=len(expected['rates']),audit_contrasts=len(expected['contrasts']))


def secondary_effects():
    languages=['pt','en','es','zh','hi'];contracts=read(ROOT/'paper/data/primary_contracts.json')
    language_targets=read(ROOT/'paper/data/language_effects.json')['rows']
    equivalence=read(ROOT/'paper/data/equivalence_margin.json')['rows']
    for key in ['source_m1','source_m3','source_m5']:
        with gzip.open(ROOT/'paper/data/response_metrics'/(key+'.json.gz'),'rt') as f:raw=json.load(f)
        groups=defaultdict(dict)
        for row in raw['observations']:
            component,context,condition,*ident=row['identity']
            groups[component,context,condition][tuple(ident)]=row
        aliases={r['id']:r['alias_of'] for r in contracts[key]['zero_dose_aliases']}
        def sample(component,context,condition):
            actual=aliases.get(condition,condition)
            if component=='languages' and actual=='baseline':component='main'
            return groups[component,context,actual]
        for target in [r for r in language_targets if key=='source_'+r['slot']]:
            context=target['context'];sign='plus' if context=='ethical' else 'minus'
            base=sample('main',context,'baseline')
            vectors={lang:sample('languages',context,target['estimator']+'_'+lang+'_'+sign) for lang in languages}
            diagonal=[];off=[]
            for source in languages:
                for dest in languages:
                    ids=[i for i in base if i[1]==dest and base[i]['semantic_unsafe'] is not None and vectors[source][i]['semantic_unsafe'] is not None]
                    delta=np.mean([vectors[source][i]['semantic_unsafe']-base[i]['semantic_unsafe'] for i in ids])*100
                    (diagonal if source==dest else off).append(delta)
            got=dict(diagonal_mean_cell_delta_pp=np.mean(diagonal),off_diagonal_mean_cell_delta_pp=np.mean(off),
                     min_off_diagonal_delta_pp=min(off),max_off_diagonal_delta_pp=max(off))
            check(got,target,got);values=[];families=[]
            for identity,row in base.items():
                outcomes=[vectors[l][identity]['semantic_unsafe'] for l in languages]
                if any(y is None for y in outcomes):continue
                same=outcomes[languages.index(identity[1])]
                values.append(same-np.mean([y for l,y in zip(languages,outcomes) if l!=identity[1]]));families.append(row['family'])
            names=tuple(sorted(set(families)));group=np.array([names.index(f) for f in families])
            ci,undefined=interval(np.bincount(group,weights=values,minlength=len(names)),np.bincount(group,minlength=len(names)),names)
            got=dict(delta_pp=np.mean(values)*100,ci95_pp=ci,paired_resolved=len(values),paired_missing=800-len(values),bootstrap_undefined=undefined)
            check(got,target['paired_same_minus_other'],got)
        for target in [r for r in equivalence if r['key']==key]:
            context=target['context'];sign='plus' if context=='ethical' else 'minus'
            a=sample('main',context,'native_pooled_'+sign);b=sample('main',context,'shadow_pooled_'+sign)
            ids=[i for i in sorted(a) if a[i]['semantic_unsafe'] is not None and b[i]['semantic_unsafe'] is not None]
            names=tuple(sorted({r['family'] for r in a.values()}));group=np.array([names.index(a[i]['family']) for i in ids])
            delta=np.array([a[i]['semantic_unsafe']-b[i]['semantic_unsafe'] for i in ids])
            ci,undefined=interval(np.bincount(group,weights=delta,minlength=len(names)),np.bincount(group,minlength=len(names)),names,tail=.01)
            got=dict(delta_pp=delta.mean()*100,interval98_pp=ci,inside_margin=ci[0]>-7 and ci[1]<7,
                     paired_resolved=len(ids),paired_missing=800-len(ids),bootstrap_undefined=undefined)
            check(got,target,got)
    return dict(language_summaries=len(language_targets),equivalence_checks=len(equivalence))


if __name__=='__main__':
    print(json.dumps(dict(status='PASS',**audit(),**secondary_effects())))
