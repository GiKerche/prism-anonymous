"""Native tool-interface generation used for Qwen and Ministral."""
import hashlib,time
from runtime import eos_ids,require,sync_all

def bfcl_batch(model,backbone,tokenizer,cases,torch):
    from prism.tool_parser import parse_tool_calls_detailed,strip_tool_call_blocks
    from prism.action_labels import action_metrics
    from prism.degeneration import diagnose_degeneration
    rendered=[]
    for c in cases:
        kwargs=dict(tools=c["tools"],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        try:prompt=tokenizer.apply_chat_template(c["question_turns"][0],**kwargs)
        except TypeError:
            kwargs.pop("enable_thinking");prompt=tokenizer.apply_chat_template(c["question_turns"][0],**kwargs)
        rendered.append(prompt)
    enc=tokenizer(rendered,return_tensors="pt",add_special_tokens=False,padding=True,truncation=False)
    lengths=enc["attention_mask"].sum(1).tolist();width=enc["input_ids"].shape[1]
    require(max(lengths)<=8192,"BFCL prompt cap")
    enc={k:v.to(backbone.embed_tokens.weight.device) for k,v in enc.items()};stops=eos_ids(tokenizer,model)
    start=time.monotonic()
    with torch.inference_mode():
        out=model.generate(**enc,max_new_tokens=2048,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=sorted(stops),use_cache=True)
    sync_all(torch);elapsed=time.monotonic()-start;rows=[]
    for i,c in enumerate(cases):
        ids=out[i,width:].tolist();stop=next((j for j,t in enumerate(ids) if t in stops),None)
        if stop is not None:ids=ids[:stop+1]
        raw=tokenizer.decode(ids,skip_special_tokens=False);parsed=parse_tool_calls_detailed(raw)
        names={t["function"]["name"] for t in c["tools"]}
        rows.append(dict(rendered_prompt_sha256=hashlib.sha256(rendered[i].encode()).hexdigest(),
            raw_response=raw,response_text=strip_tool_call_blocks(raw),generated_token_ids=ids,
            n_prompt_tokens=int(lengths[i]),n_generated_tokens=len(ids),truncated=stop is None and len(ids)>=2048,
            stop_reason="eos" if stop is not None else "length" if len(ids)>=2048 else "other",
            parse=parsed.to_dict(),action_metrics=action_metrics(parsed.calls),
            recognized_tool_schema=bool(parsed.calls) and all(c["name"] in names for c in parsed.calls),
            malformed=parsed.status=="malformed",degeneration=diagnose_degeneration(raw).to_dict(),elapsed_seconds=elapsed))
    return rows,elapsed
