"""Reproduce fixed-adapter contrasts and capability from portable endpoints.

Python 3 and NumPy only. No generation, judging, or model fitting.
"""
from collections import defaultdict
from functools import lru_cache
import gzip
import json
from pathlib import Path
import numpy as np
from family_statistics import draws, rate, paired

ROOT = Path(__file__).resolve().parents[1]
REPLICAS = ['lora_26082701', 'lora_26082702', 'lora_26082703']


def read(name):
    path = ROOT / 'data' / name
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def equal(actual, expected, fields):
    for key in fields:
        if expected[key] is None:
            assert actual[key] is None, key
        else:
            assert np.allclose(actual[key], expected[key], atol=1e-9, rtol=0), (key, actual[key], expected[key])


def mean_change(lefts, right, families, multiplicity=1):
    names = tuple(sorted(set(families)))
    group = np.array([names.index(f) for f in families])
    weights = draws(names)
    estimates, samples = [], []
    for left in lefts:
        estimates.append(paired(left, right, families)['delta_pp'])
        a = np.asarray([np.nan if x is None else x for x in left],dtype=float)
        b = np.asarray([np.nan if x is None else x for x in right],dtype=float)
        valid = np.isfinite(a) & np.isfinite(b)
        num = np.bincount(group[valid], weights=(a-b)[valid], minlength=len(names))
        den = np.bincount(group[valid], minlength=len(names))
        denominator = weights @ den
        samples.append(np.divide(weights @ num, denominator, out=np.full(10000, np.nan), where=denominator>0))
    distribution = 100*np.mean(samples, axis=0)
    distribution = distribution[np.isfinite(distribution)]
    tail = .025/multiplicity
    return dict(delta_pp=float(np.mean(estimates)), ci95_pp=np.quantile(distribution, [.025, .975]),
        bonferroni_ci_pp=np.quantile(distribution, [tail, 1-tail]) if multiplicity>1 else None,
        replica_delta_sd_pp=float(np.std(estimates, ddof=1)))


def main():
    report = read('current_lora_results.json')
    groups = defaultdict(dict)
    for row in read('response_metrics/lora.json.gz'):
        ident = (row['scenario_id'], row['language'], row['seed'])
        group = groups[row['key'], row['component'], row['context']]
        assert ident not in group
        group[ident] = row
    qwen = read('response_metrics/source_m1.json.gz')
    for row in qwen['observations']:
        component, context, cid, scenario, language, seed = row['identity']
        groups[cid, component, context][scenario, language, seed] = row

    @lru_cache(maxsize=None)
    def sample(cid, component, context, language):
        return {i: row for i, row in groups[cid, component, context].items()
                if language == 'all' or i[1] == language}

    for target in report['rates']:
        rows = sample(target['condition'], target['component'], target['context'], target['target_language'])
        ids = sorted(rows)
        got = rate([rows[i][target['metric']] for i in ids], [rows[i]['family'] for i in ids])
        equal(got, target, ['positive', 'resolved', 'missing', 'rate_percent', 'ci95_percent'])
    for target in report['contrasts']:
        args = target['component'], target['context'], target['target_language']
        right = sample(target['right'], *args)
        lefts = [sample(cid, *args) for cid in REPLICAS]
        ids = sorted(right)
        assert all(set(left) == set(right) for left in lefts)
        metric = target['metric']
        got = mean_change([[left[i][metric] for i in ids] for left in lefts],
            [right[i][metric] for i in ids], [right[i]['family'] for i in ids], target['multiplicity'])
        equal(got, target, ['delta_pp', 'ci95_pp', 'bonferroni_ci_pp', 'replica_delta_sd_pp'])

    benign = defaultdict(dict)
    for row in read('response_metrics/lora_bfcl.json.gz'):
        benign[row['condition']][row['bfcl_id']] = row
    for row in qwen['controls']:
        cid, bid = row['identity']
        benign[cid][bid] = row
    base = benign['baseline']
    ids = sorted(base)
    assert len(ids) == 914
    families = [base[i]['family'] for i in ids]
    for target in report['benign']:
        assert target['stratum'] == 'tool_required'
        metric, cid = target['metric'], target['condition']
        if cid == 'mean_lora':
            got = mean_change([[benign[c][i][metric] for i in ids] for c in REPLICAS],
                [base[i][metric] for i in ids], families)
            equal(got, target['paired_vs_baseline'], ['delta_pp', 'ci95_pp', 'replica_delta_sd_pp'])
        else:
            assert set(benign[cid]) == set(base)
            values = [benign[cid][i][metric] for i in ids]
            got = rate(values, families)
            equal(got, target, ['positive', 'resolved', 'rate_percent', 'ci95_percent'])
            if 'paired_vs_baseline' in target:
                got = paired(values, [base[i][metric] for i in ids], families)
                equal(got, target['paired_vs_baseline'], ['delta_pp', 'ci95_pp'])
    print(json.dumps(dict(status='LORA_STATISTICS_REPRODUCED', rates=len(report['rates']),
        contrasts=len(report['contrasts']), capability_rows=len(report['benign']), judge_accuracy_validated=False)))


if __name__ == '__main__':
    main()
