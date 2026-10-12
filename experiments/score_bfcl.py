"""Score fresh run_bfcl.py outputs with the pinned official BFCL checker."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import bfcl_checker
from runtime import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True,help='A run_bfcl output directory')
    parser.add_argument('--benchmark-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/bfcl_scored.json')
    args=parser.parse_args()
    bfcl_checker.OFFICIAL_ROOT=args.benchmark_root.resolve()
    expected=json.loads((ROOT/'experiments/bfcl_checker_hashes.json').read_text())
    for rel,digest in expected.items():
        if hashlib.sha256((args.benchmark_root/rel).read_bytes()).hexdigest()!=digest:
            raise ValueError('BFCL checker snapshot differs: '+rel)
    originals={};golds={}
    for partition in ['test','validation_ii']:
        for suffix,dest in [('original',originals),('gold',golds)]:
            rows=json.loads((ROOT/'experiments/inputs'/f'bfcl_{partition}_{suffix}.json').read_text())['cases']
            dest[partition]={r['bfcl_id']:r for r in rows}
    results=defaultdict(list);seen=set()
    for path in sorted(args.input.rglob('batch_*.json')):
        batch=json.loads(path.read_text());recipe=batch['recipe']
        key=(recipe['model'],recipe['partition'],recipe['actual_condition'])
        for row in batch['rows']:
            case,generation=row['case'],row['generation'];bid=case['bfcl_id']
            if key+(bid,) in seen:raise ValueError('Duplicate response in scoring input')
            seen.add(key+(bid,))
            error=None
            if recipe['interface']=='Prompt/Python':
                raw=re.sub(r'(?:<\|eom_id\|>|<\|eot_id\|>|<\|end_of_text\|>)+\s*$','',generation['raw_response'])
                try:
                    decoded=bfcl_checker.load_prompt_decoder()(raw)
                    calls=[dict(name=n,arguments=p) for item in decoded for n,p in item.items()]
                except Exception as exc:
                    calls=[];error=type(exc).__name__
            else:
                from prism.tool_parser_bfcl_v3 import parse_bfcl_tool_calls
                parsed=parse_bfcl_tool_calls(generation['raw_response'],case['tools'],recipe['model'].split('_')[1])
                calls=parsed.calls
                if parsed.status=='malformed':error='malformed'
            scored=bfcl_checker.evaluate_item(originals[key[1]][bid],golds[key[1]][bid],calls)
            results[key].append(dict(bfcl_id=bid,official_valid=bool(scored['official_valid'] and error is None),decoder_error=error))
    if not results:raise ValueError('No BFCL generation batches found')
    arms=[]
    for (model,partition,condition),rows in sorted(results.items()):
        complete={r['bfcl_id'] for r in rows}==set(originals[partition])
        arms.append(dict(model=model,partition=partition,condition=condition,complete=complete,
                         total=len(rows),correct=sum(r['official_valid'] for r in rows),rows=rows))
    atomic_json(args.output,dict(arms=arms))
    for arm in arms:print(json.dumps({k:v for k,v in arm.items() if k!='rows'}))


if __name__=='__main__':main()
