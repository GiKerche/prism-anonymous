"""Run fixed source interventions on the pinned simulated SODA environments."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path
import numpy as np
from runtime import load_model,Intervention,atomic_json
from soda_inputs import load_tasks

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=['source_m1','source_m3','source_m5'])
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--benchmark-root',type=Path)
    parser.add_argument('--dataset',type=Path,help='Pinned SODA full_interaction test Parquet, to reconstruct the complete inputs in memory.')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/soda')
    parser.add_argument('--direction',choices=['native_pooled','shadow_pooled'],default='shadow_pooled')
    parser.add_argument('--sign',choices=['plus','minus'],default='plus')
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    spec=json.loads((ROOT/'reproduction/protocol.json').read_text())['models'][args.model]
    tasks,pending=load_tasks(args.dataset,require_complete=not args.dry_run);assert len(tasks)==400
    dose=0. if args.baseline else spec['selected_alphas'][args.direction.split('_')[0]+'_'+args.sign]
    if isinstance(dose,dict):dose=dose['alpha']
    condition='baseline' if dose==0 else args.direction+'_'+args.sign
    recipe=dict(model=args.model,revision=spec['model_revision'],layer=spec['layer'],a=dose,sigma=spec['sigma'],
        condition=condition,tasks=400,batch_size=4,max_new_tokens=2048,do_sample=False,max_rounds_per_turn=5,
        external_fixtures_required=pending)
    print(json.dumps(recipe,indent=2))
    if args.dry_run:return
    assert args.checkpoint is not None and args.benchmark_root is not None
    manifest=json.loads((ROOT/'experiments/soda_environment_manifest.json').read_text())
    for item in manifest:
        path=args.benchmark_root/item['path']
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],item['path']
    sys.path.insert(0,str(args.benchmark_root))
    registry=importlib.import_module('envs').ENV_REGISTRY
    import torch,soda_runtime
    role,slot=args.model.split('_');soda_runtime.MODEL_FAMILY=slot
    model,tokenizer,backbone,_=load_model(contract=json.loads((ROOT/'experiments/models.json').read_text()),
        slot=slot.upper(),role=role,model_root=args.checkpoint,torch=torch)
    vectors=np.load(ROOT/'reproduction'/args.model/'directions.npz',allow_pickle=False)
    intervention=None if dose==0 else Intervention(torch=torch,layer=backbone.layers[spec['layer']],
        vector=torch.from_numpy(vectors[args.direction].copy()),strength=dose*spec['sigma'])
    for start in range(0,len(tasks),4):
        path=args.output/args.model/condition/f'batch_{start//4:04d}.json'
        if path.exists():
            assert json.loads(path.read_text())['recipe']==recipe,'Existing output uses a different recipe'
            continue
        outputs,hooks=soda_runtime.run_group(tasks=tasks[start:start+4],registry=registry,torch=torch,
            model=model,tokenizer=tokenizer,intervention=intervention,max_new_tokens=2048,max_rounds_per_turn=5)
        atomic_json(path,dict(recipe=recipe,trajectories=outputs,hooks=hooks))

if __name__=='__main__':main()
