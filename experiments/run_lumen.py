"""Portable BF16 generation using the final model loader and steering hook.

Run --dry-run to inspect the exact inputs, dose and recipe without model access.
New stochastic generations are replications, not replacements for published data.
"""
import argparse
from pathlib import Path
import json
import numpy as np
from runtime import load_model, generate_batch, intervention_context, atomic_json

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=['source_m1','source_m3','source_m5','receiver_m1','receiver_m3','receiver_m5'])
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/lumen')
    parser.add_argument('--partition',choices=['build','layer_selection','alpha_selection','final_test'],default='final_test')
    parser.add_argument('--context',choices=['ethical','neutral'],default='ethical')
    parser.add_argument('--direction',default='native_pooled')
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--language',choices=['pt','en','es','zh','hi','ar'])
    parser.add_argument('--dose',type=float,help='Only for explicit new replication of the published Validation II grid.')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    config=json.loads((ROOT/'reproduction/protocol.json').read_text())['models'][args.model]
    contract=json.loads((ROOT/'experiments/models.json').read_text())
    cases=json.loads((ROOT/'paper/task/scenario_inputs.json').read_text())['cases']
    cases=[c for c in cases if c['partition']==args.partition and c['context']==args.context
           and (c['language']==args.language if args.language else c['language']!='ar')]
    assert cases and args.batch_size>0
    if args.language=='ar':assert args.partition=='final_test','Arabic supplies no fitting/tuning data'
    sign='plus' if args.context=='ethical' else 'minus'
    direction=args.direction
    if args.model.startswith('receiver') and direction=='native_pooled':direction='transported_native'
    with np.load(ROOT/'reproduction'/args.model/'directions.npz',allow_pickle=False) as saved:
        assert direction in saved.files,'Direction unavailable for this model'
    if args.baseline:dose=0.
    elif args.dose is not None:
        assert args.partition=='alpha_selection','New dose selection uses Validation II only'
        assert abs(args.dose) in [.25,.5,1,2,4,8]
        assert (args.dose>0)==(sign=='plus');dose=args.dose
    elif direction=='receiver_native':
        controls=json.loads((ROOT/'paper/data/current_receiver_controls.json').read_text())
        dose=controls['selections'][args.model]['selected_alphas']['receiver_native_'+sign]['alpha']
    else:
        estimator='transported_native' if args.model.startswith('receiver') else ('shadow' if direction.startswith('shadow_') else 'native')
        doses=config['selected_alphas'];dose=doses[estimator+'_'+sign]
        if isinstance(dose,dict):dose=dose['alpha']
    seeds=list(range(101,133)) if args.partition=='layer_selection' else [101,102,103,104]
    if args.partition in ['build','layer_selection']:assert dose==0,'Fitting data must be unsteered'
    logical='baseline' if args.baseline else direction+'_'+sign
    actual='baseline' if dose==0 else logical
    if args.dose is not None and dose!=0:actual+='_a'+str(abs(dose)).replace('.','p')
    recipe=dict(model=args.model,model_id=config['model_id'],revision=config['model_revision'],partition=args.partition,
        context=args.context,layer=config['layer'],direction=direction,logical_condition=logical,actual_condition=actual,
        a=dose,sigma=config['sigma'],temperature=.7,top_p=.95,max_new_tokens=1024,
        seeds=seeds,batch_size=args.batch_size,inputs=len(cases),qwen_enable_thinking=False if args.model.endswith('m1') else None)
    print(json.dumps(recipe,indent=2))
    if args.dry_run:return
    assert args.checkpoint is not None,'Supply the pinned local checkpoint directory'
    import torch
    from prism.tool_parser_action_v4 import parse_action_calls_v4
    from prism.action_labels import action_metrics
    role,slot=args.model.split('_')
    model,tokenizer,backbone,_=load_model(contract=contract,slot=slot.upper(),role=role,model_root=args.checkpoint,torch=torch)
    vectors=np.load(ROOT/'reproduction'/args.model/'directions.npz',allow_pickle=False)
    vector=None if dose==0 else torch.from_numpy(vectors[direction].copy())
    tools=json.loads((ROOT/'paper/task/tools_by_language.json').read_text())
    if 'tools_by_language' in tools:tools=tools['tools_by_language']
    args.output.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output/(logical+'.recipe.json'),recipe)
    for language in ['pt','en','es','zh','hi','ar']:
        group=[c for c in cases if c['language']==language]
        for seed in seeds:
            for start in range(0,len(group),args.batch_size):
                path=args.output/args.model/args.partition/args.context/actual/language/f'seed_{seed}_batch_{start//args.batch_size:04d}.json'
                if path.exists():
                    assert json.loads(path.read_text())['recipe']==recipe,'Existing output uses a different recipe'
                    continue
                batch=group[start:start+args.batch_size]
                with intervention_context(torch=torch,layer=backbone.layers[config['layer']],vector=vector,strength=dose*config['sigma']):
                    outputs,_=generate_batch(model=model,backbone=backbone,tokenizer=tokenizer,cases=batch,
                        tools_by_language=tools,seed=seed,torch=torch)
                for case,response in zip(batch,outputs):
                    parsed=parse_action_calls_v4(response['raw_response'],tools[language],slot)
                    response['corrected_parse']=parsed.to_dict();response['corrected_action_metrics']=action_metrics(parsed.calls)
                atomic_json(path,dict(recipe=recipe,cases=batch,responses=outputs))

if __name__=='__main__':main()
