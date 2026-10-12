"""Fixed family bootstrap for response rates and paired final-test contrasts."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=64)
def draws(families, replicates=10000, seed=20260928):
    n=len(families)
    if not n:raise ValueError('No families')
    return np.random.default_rng(seed).multinomial(n,np.full(n,1/n),size=replicates)


def values_and_groups(values,families):
    if len(values)!=len(families) or not len(values):raise ValueError('Empty or unpaired data')
    if any(v is not None and v not in [0,1,False,True] for v in values):raise ValueError('Nonbinary outcome')
    names=tuple(sorted(set(families)));group=np.array([names.index(f) for f in families])
    return np.array([np.nan if v is None else v for v in values],dtype=float),names,group


def interval(numerators,denominators,names,tail=.025):
    counts=draws(names);den=counts@denominators;valid=den>0
    boot=(counts@numerators)[valid]/den[valid]
    return (np.quantile(boot,[tail,1-tail])*100).tolist() if len(boot) else None,int((~valid).sum())


def rate(values,families):
    data,names,group=values_and_groups(values,families);valid=np.isfinite(data)
    n=len(data);resolved=int(valid.sum());positive=int(np.nansum(data));missing=n-resolved
    numer=np.bincount(group[valid],weights=data[valid],minlength=len(names))
    denom=np.bincount(group[valid],minlength=len(names))
    ci,undefined=interval(numer,denom,names)
    return dict(positive=positive,resolved=resolved,total=n,missing=missing,
                rate_percent=positive/resolved*100 if resolved else None,coverage_percent=resolved/n*100,
                all_response_bounds_percent=[positive/n*100,(positive+missing)/n*100],
                ci95_percent=ci,bootstrap_undefined=undefined,families=len(names),
                degenerate_interval=bool(ci and ci[0]==ci[1]),
                uncertainty='Family bootstrap conditional on observed judge labels; no judge-error or zero-risk claim')


def paired(left,right,families,multiplicity=1,same_observations=False):
    a,names,group=values_and_groups(left,families);b,other,_=values_and_groups(right,families)
    if names!=other:raise ValueError('Family mismatch')
    if same_observations and not np.array_equal(a,b,equal_nan=True):raise ValueError('Alias differs from original')
    valid=np.isfinite(a)&np.isfinite(b);delta=a[valid]-b[valid]
    den=np.bincount(group[valid],minlength=len(names))
    num=np.bincount(group[valid],weights=delta,minlength=len(names))
    ci,undefined=interval(num,den,names)
    adjusted,_=interval(num,den,names,tail=.025/multiplicity)
    lower=np.where(np.isfinite(a),a,0)-np.where(np.isfinite(b),b,1)
    upper=np.where(np.isfinite(a),a,1)-np.where(np.isfinite(b),b,0)
    return dict(delta_pp=float(delta.mean()*100) if len(delta) else None,paired_resolved=int(valid.sum()),
        paired_total=len(a),paired_missing=int((~valid).sum()),ci95_pp=ci,
        multiplicity=multiplicity,bonferroni_ci_pp=adjusted if multiplicity>1 else None,
        all_response_delta_bounds_pp=[0.,0.] if same_observations else [float(lower.mean()*100),float(upper.mean()*100)],
        bootstrap_undefined=undefined,exact_baseline_alias=same_observations,
        degenerate_interval=bool(ci and ci[0]==ci[1]),families=len(names))
