"""Verify browser responses against deposited endpoints and content hashes."""
from pathlib import Path
import collections
import gzip
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
def gz(path):return json.loads(gzip.decompress(path.read_bytes()))
manifest=json.loads((ROOT/'docs/data/responses/manifest.json').read_text())
control_rows=gz(ROOT/'paper/data/response_metrics/receiver_controls.json.gz')
control_index={(r['key'],r['context'],r['condition'],*r['identity']):r['value'] for r in control_rows}
models=collections.defaultdict(list)
checked=0
for name,entry in manifest['shards'].items():
    path=ROOT/'docs'/entry['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],name
    rows=gz(path)
    assert len(rows)==entry['responses'],name
    for r in rows:
        assert hashlib.sha256(r['response'].encode()).hexdigest()==r.get('display_sha256',r['response_sha256'])
        if r.get('display_redacted'):assert r.get('redaction_note') and r.get('display_sha256')
        assert r['semantic_unsafe'] in (None,0,1)
    models[name.split('/')[0]].extend(rows)

for key,rows in models.items():
    original=gz(ROOT/'paper/data/response_metrics'/f'{key}.json.gz')['observations']
    index={tuple(r['identity']):r for r in original}
    identities=set()
    for r in rows:
        identity=tuple(r[k]for k in ['component','context','condition','scenario_id','language','seed'])
        assert identity not in identities,(key,identity)
        identities.add(identity)
        if r['condition'].startswith('selected_receiver_native'):
            ci=(key,r['context'],r['condition'],r['scenario_id'],r['language'],r['seed'])
            assert ci in control_index,ci
            assert control_index[ci]==r['semantic_unsafe'],ci
        else:
            expected=index[identity]
            for field in ['semantic_unsafe','call_proxy','truncated','family']:
                assert r[field]==expected[field],(key,identity,field)
        checked+=1

    # Verify all displayed primary rates, including per-language and zero-dose aliases.
    published=json.loads((ROOT/'docs/data/catalog.json').read_text())['results'][key]['rates']
    for rate in published:
        condition=rate['actual_condition']
        comp=rate['component']
        if condition=='baseline' and comp=='languages':comp='main'
        selected=[r for r in rows if r['condition']==condition and r['context']==rate['context']
                  and r['component']==comp and
                  (rate['target_language']=='all' or r['language']==rate['target_language'])]
        assert len(selected)==rate['total'],(key,rate['condition'],comp,len(selected),rate['total'])
        assert sum(r['semantic_unsafe']==1 for r in selected)==rate['positive']
        assert sum(r['semantic_unsafe'] is not None for r in selected)==rate['resolved']
print(json.dumps({'responses_verified':checked,'models':len(models),'shards':len(manifest['shards']),
                  'primary_rates_match':True,'display_hashes_match':True,
                  'documented_display_redactions':manifest.get('display_redactions',{}).get('responses',0)}))
