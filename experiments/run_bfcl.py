"""Generate the frozen BFCL partition with each model's final tool interface."""
import argparse,json
from pathlib import Path
import numpy as np
from runtime import load_model,intervention_context,atomic_json

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,choices=['source_m1','source_m3','source_m5','receiver_m1','receiver_m3','receiver_m5'])
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/bfcl')
    p.add_argument('--partition',choices=['test','validation_ii'],default='test')
    p.add_argument('--direction',default='native_pooled')
    p.add_argument('--sign',choices=['plus','minus'],default='plus')
    p.add_argument('--baseline',action='store_true')
    p.add_argument('--dose',type=float)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    spec=json.loads((ROOT/'reproduction/protocol.json').read_text())['models'][args.model]
    role,slot=args.model.split('_');llama=slot=='m5'
    inputs=ROOT/'experiments/inputs'/f'bfcl_{args.partition}_{"llama" if llama else args.model}.json'
    cases=json.loads(inputs.read_text())['cases']
    assert len(cases)==(914 if args.partition=='test' else 86)
    direction=args.direction
    if role=='receiver' and direction=='native_pooled':direction='transported_native'
    with np.load(ROOT/'reproduction'/args.model/'directions.npz',allow_pickle=False) as saved:
        assert direction in saved.files,'Direction unavailable for this model'
    if args.baseline:dose=0.
    elif args.dose is not None:
        assert args.partition=='validation_ii' and abs(args.dose) in [.25,.5,1,2,4,8]
        assert (args.dose>0)==(args.sign=='plus');dose=args.dose
    elif direction=='receiver_native':
        ctrl=json.loads((ROOT/'paper/data/current_receiver_controls.json').read_text())
        dose=ctrl['selections'][args.model]['selected_alphas']['receiver_native_'+args.sign]['alpha']
    else:
        estimator='transported_native' if role=='receiver' else ('shadow' if direction.startswith('shadow_') else 'native')
        dose=spec['selected_alphas'][estimator+'_'+args.sign]
        if isinstance(dose,dict):dose=dose['alpha']
    actual='baseline' if dose==0 else direction+'_'+args.sign
    if args.dose is not None and dose!=0:actual+='_a'+str(abs(dose)).replace('.','p')
    recipe=dict(model=args.model,model_id=spec['model_id'],revision=spec['model_revision'],partition=args.partition,
        interface='Prompt/Python' if llama else 'native tools',layer=spec['layer'],sigma=spec['sigma'],a=dose,
        direction=direction,actual_condition=actual,do_sample=False,max_new_tokens=2048,batch_size=4,cases=len(cases))
    print(json.dumps(recipe,indent=2))
    if args.dry_run:return
    assert args.checkpoint is not None
    import torch
    from bfcl_prompt import generate
    from bfcl_native import bfcl_batch
    from prism.tool_parser_bfcl_v3 import parse_bfcl_tool_calls
    model,tokenizer,backbone,_=load_model(contract=json.loads((ROOT/'experiments/models.json').read_text()),
        slot=slot.upper(),role=role,model_root=args.checkpoint,torch=torch)
    arrays=np.load(ROOT/'reproduction'/args.model/'directions.npz',allow_pickle=False)
    vector=None if dose==0 else torch.from_numpy(arrays[direction].copy())
    for category in sorted({c['category'] for c in cases}):
        group=[c for c in cases if c['category']==category]
        for start in range(0,len(group),4):
            path=args.output/args.model/args.partition/actual/category/f'batch_{start//4:04d}.json'
            if path.exists():
                assert json.loads(path.read_text())['recipe']==recipe,'Existing output uses a different recipe'
                continue
            batch=group[start:start+4]
            with intervention_context(torch=torch,layer=backbone.layers[spec['layer']],vector=vector,strength=dose*spec['sigma']):
                if llama:rows,_=generate(model,backbone,tokenizer,batch,torch)
                else:
                    values,_=bfcl_batch(model,backbone,tokenizer,batch,torch)
                    rows=[dict(case=c,generation=g) for c,g in zip(batch,values)]
            for row in rows:
                if not llama:
                    parsed=parse_bfcl_tool_calls(row['generation']['raw_response'],row['case']['tools'],slot)
                    row['corrected_parse']=parsed.to_dict()
                # Llama Prompt/Python is decoded by the pinned official BFCL
                # prompting decoder during scoring, not by a native-call parser.
            atomic_json(path,dict(recipe=recipe,rows=rows))

if __name__=='__main__':main()
