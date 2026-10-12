"""Train a fresh replica with the fixed construction-only LoRA recipe."""
import argparse
import json
import math
import random
from pathlib import Path
from runtime import load_model
from lora_training import tokenize_examples,collate,groups_for_epoch

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/lora')
    parser.add_argument('--seed',type=int,choices=[26082701,26082702,26082703],required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    p=json.loads((ROOT/'reproduction/lora/protocol.json').read_text())
    rows=json.loads((ROOT/'reproduction/lora/construction_targets.json').read_text())
    split=json.loads((ROOT/'paper/task/partitions.json').read_text())
    assert len(rows)==55 and all(r['scenario_id'] in split['roles']['build'] and r['split']=='build' for r in rows)
    print(json.dumps(dict(seed=args.seed,construction_targets=len(rows),lora=p['lora'],optimization=p['optimization']),indent=2))
    if args.dry_run:return
    assert args.checkpoint is not None
    import torch
    from safetensors.torch import save_file
    from prism.lora import inject_all_backbone_linears,adapter_parameters,adapter_state_dict
    model,tokenizer,_,_=load_model(contract=json.loads((ROOT/'experiments/models.json').read_text()),slot='M1',role='source',model_root=args.checkpoint,torch=torch)
    seed=args.seed;random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    lp=p['lora'];injection=inject_all_backbone_linears(model,rank=lp['rank'],alpha=lp['alpha'],dropout=lp['dropout'])
    assert len(injection.module_names)==248 and injection.trainable_parameters==43278336
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.train();tokenizer.padding_side='right'
    examples,_=tokenize_examples(tokenizer,rows)
    opt=p['optimization'];params=list(adapter_parameters(model));planned=opt['total_optimizer_steps']
    optimizer=torch.optim.AdamW(params,lr=opt['learning_rate'],weight_decay=opt['weight_decay'],fused=True)
    def lr_multiplier(step):
        warmup=opt['warmup_optimizer_steps']
        if step<warmup:return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,planned-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_multiplier)
    losses=[]
    for epoch in range(opt['epochs']):
        groups=groups_for_epoch(len(rows),seed=seed,epoch=epoch,micro_batch_size=opt['micro_batch_size'],gradient_accumulation_steps=opt['gradient_accumulation_steps'])
        assert [sum(map(len,g)) for g in groups]==[16,16,16,7]
        for group in groups:
            optimizer.zero_grad(set_to_none=True)
            batches=[collate(torch,examples,indices,tokenizer.pad_token_id) for indices in group]
            total=sum(b['completion_tokens'] for b in batches);group_loss=0.
            for batch in batches:
                tokens=batch.pop('completion_tokens');batch={k:v.cuda() for k,v in batch.items()}
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):loss=model(**batch,use_cache=False).loss
                assert torch.isfinite(loss);weight=tokens/total;(loss*weight).backward()
                group_loss+=float(loss.detach().cpu())*weight
                del loss,batch
            norm=torch.nn.utils.clip_grad_norm_(params,opt['max_grad_norm']);assert torch.isfinite(norm)
            optimizer.step();scheduler.step();losses.append(group_loss)
            print('step',len(losses),'loss',group_loss,flush=True)
    assert len(losses)==planned==20
    args.output.mkdir(parents=True,exist_ok=True)
    dest=args.output/f'adapter_{seed}.safetensors';assert not dest.exists()
    save_file(adapter_state_dict(model),str(dest))
    (args.output/f'train_{seed}.json').write_text(json.dumps(dict(seed=seed,losses=losses,protocol=p),indent=2)+'\n')

if __name__=='__main__':main()
