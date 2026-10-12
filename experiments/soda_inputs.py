"""Restore omitted benchmark fixtures in memory from the pinned SODA data.

The public repository never stores the credential-shaped fixture. This loader
does not contact a service, authenticate, or write the restored input to disk.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_tasks(dataset=None, *, require_complete=True):
    bundle = json.loads((ROOT / 'experiments/inputs/soda_tasks.json').read_text())
    needs_dataset = bundle.get('requires_external_dataset', False)
    if needs_dataset and dataset is None:
        if require_complete:
            raise ValueError('Fresh SODA execution requires --dataset with the pinned '
                             'full_interaction test Parquet; the task manifest '
                             'does not contain generation inputs.')
        return bundle['tasks'], len(bundle['tasks'])
    if needs_dataset:
        dataset = Path(dataset)
        pin = json.loads((ROOT / 'experiments/benchmarks.json').read_text())['soda']
        if hashlib.sha256(dataset.read_bytes()).hexdigest() != pin['dataset_sha256']:
            raise ValueError('SODA dataset checksum differs from the recorded snapshot')
        import pyarrow.parquet as pq
        rows = {str(row['id']): row for row in pq.read_table(dataset).to_pylist()}
        tasks = []
        for preview in bundle['tasks']:
            row = rows[preview['id']]
            tasks.append(dict(
                id=str(row['id']), scenario=str(row['scenario']),
                threat_depth=int(row['threat_depth']),
                user_turns=json.loads(row['user_turns']),
                prefilled_messages=json.loads(row['prefilled_messages']),
                environments=[dict(name=str(row['env_name']),
                    parameters=json.loads(row['parameters']), tools=json.loads(row['tools']))],
            ))
        serialized = (json.dumps(tasks, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(',', ':')) + '\n').encode()
        if hashlib.sha256(serialized).hexdigest() != bundle['tasks_sha256']:
            raise ValueError('Reconstructed SODA tasks differ from the frozen generation inputs')
        bundle['tasks'] = tasks
    return bundle['tasks'], 0
