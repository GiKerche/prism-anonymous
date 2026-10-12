"""Reproduce added semantic contrasts from portable response-level endpoints.

Requires Python 3 and NumPy. Does not call a model or change any labels.
This validates statistical reproduction, not semantic-judge accuracy.
"""
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from family_statistics import paired, rate

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT/'data'/name).read_text())


def close(actual, expected):
    assert np.allclose(actual, expected, atol=1e-9, rtol=0), (actual, expected)


def controls(kind='source'):
    data = read('current_'+kind+'_controls.json')
    with gzip.open(ROOT/('data/response_metrics/'+kind+'_controls.json.gz'), 'rt') as f:
        records = json.load(f)
    arms = defaultdict(dict)
    for r in records:
        key = (r['key'], r['context'], r['condition'])
        ident = tuple(r['identity'])
        assert ident not in arms[key]
        arms[key][ident] = r
    for a in data['arms']:
        rows = arms[a['key'], a['context'], a['condition']]
        ids = sorted(rows)
        calc = rate([rows[i]['value'] for i in ids], [rows[i]['family'] for i in ids])
        for key in ['positive', 'resolved', 'missing', 'rate_percent', 'ci95_percent']:
            close(calc[key], a[key])
    for c in data['contrasts']:
        a = arms[c['key'], c['context'], c['condition']]
        b = arms[c['key'], c['context'], c['reference']]
        assert set(a) == set(b) and len(a) == 800
        ids = sorted(a)
        assert all(a[i]['family'] == b[i]['family'] for i in ids)
        calc = paired([a[i]['value'] for i in ids], [b[i]['value'] for i in ids],
                      [a[i]['family'] for i in ids], multiplicity=data['multiplicity'],
                      same_observations=c.get('exact_baseline_alias', False))
        for key in ['delta_pp', 'paired_resolved', 'paired_missing', 'ci95_pp',
                    'bonferroni_ci_pp', 'all_response_delta_bounds_pp']:
            close(calc[key], c[key])
    return len(data['arms']), len(data['contrasts'])


def control_capability(kind):
    data = read('current_'+kind+'_controls.json')
    with gzip.open(ROOT/('data/response_metrics/'+kind+'_controls_bfcl.json.gz'), 'rt') as f:
        records = json.load(f)
    arms = defaultdict(dict)
    for r in records:
        key = (r['key'], r['condition'])
        assert r['identity'] not in arms[key]
        arms[key][r['identity']] = r
    for a in data['arms']:
        left = arms[a['key'], a['condition']]
        right = arms[a['key'], 'baseline']
        assert set(left) == set(right) and len(left) == 914
        ids = sorted(left)
        assert all(left[i]['family'] == right[i]['family'] for i in ids)
        values = [left[i]['value'] for i in ids]
        families = [left[i]['family'] for i in ids]
        rr = rate(values, families)
        expected = a['capability']
        change = paired(values, [right[i]['value'] for i in ids], families,
                        same_observations=expected['change'].get('exact_baseline_alias', False))
        for k in ['positive', 'resolved', 'missing', 'rate_percent', 'ci95_percent']:
            close(rr[k], expected['rate'][k])
        for k in ['delta_pp', 'paired_resolved', 'paired_missing', 'ci95_pp']:
            close(change[k], expected['change'][k])
    return len(data['arms'])


def soda():
    data = read('current_soda_results.json')
    with gzip.open(ROOT/'data/response_metrics/soda_trajectories.json.gz', 'rt') as f:
        records = json.load(f)
    assert len(records) == 6400
    arms = defaultdict(dict)
    for r in records:
        k = (r['key'], r['condition'])
        assert r['task_id'] not in arms[k]
        arms[k][r['task_id']] = r
    for a in data['arms']:
        rows = arms[a['key'], a['condition']['id']]
        values = [r['value'] for r in rows.values() if r['value'] is not None]
        assert len(rows) == 400
        close(len(values), a['valid']); close(sum(values), a['unsafe'])
        close(100*sum(values)/len(values), a['unsafe_percent'])
    replicas = [arms[k, k] for k in ['lora_26082701', 'lora_26082702', 'lora_26082703']]
    means = {}
    for tid, r in replicas[0].items():
        values = [a[tid]['value'] for a in replicas]
        means[tid] = dict(r, value=None if any(v is None for v in values) else float(np.mean(values)))
    checked = 0
    for c in data['contrasts']:
        if c.get('alias'):
            assert c['key'] == 'source_m1' and c['condition'].endswith('minus')
            continue
        key = c['key']
        a = means if key == 'lora_mean' else arms[key, key if key.startswith('lora_') else c['condition']]
        b = arms['source_m1' if key.startswith('lora_') else key, c['reference']]
        grouped = defaultdict(list); valid = 0
        for tid in sorted(b):
            assert a[tid]['scenario'] == b[tid]['scenario']
            if a[tid]['value'] is not None and b[tid]['value'] is not None:
                grouped[b[tid]['scenario']].append(a[tid]['value']-b[tid]['value'])
                valid += 1
        effects = np.array([np.mean(grouped[s]) for s in sorted(grouped)])
        assert len(effects) == c['scenario_clusters'] == 80
        close(valid, c['valid_pairs']); close(100*np.mean(effects), c['delta_pp'])
        rng = np.random.default_rng(c['seed'])
        distribution = 100*effects[rng.integers(0, len(effects), size=(10000, len(effects)))].mean(axis=1)
        close(np.quantile(distribution, [.025, .975]), c['ci95_pp'])
        tail = .025/c['multiplicity_family_size']
        close(np.quantile(distribution, [tail, 1-tail]), c['simultaneous_ci_pp'])
        checked += 1
    return len(data['arms']), checked


if __name__ == '__main__':
    a, c = controls(); b, d = controls('receiver'); f = control_capability('receiver'); g = control_capability('source'); s, t = soda()
    print(json.dumps(dict(status='ADDITIONAL_STATISTICS_REPRODUCED',
        control_arms=a+b, control_contrasts=c+d, receiver_capability_arms=f, source_capability_arms=g, soda_arms=s, soda_contrasts=t,
        modifies_results=False, judge_accuracy_validated=False)))
