"""Pinned Prompt/Python rendering and greedy generation for the Llama models."""
import hashlib,time
from runtime import eos_ids,require,sync_all

def render(tokenizer, case, *, flat):
    # Common BFCL system instructions; model-specific chat serialization only.
    if 'rendered_prompt_for_audit' in case:
        return case['rendered_prompt_for_audit']
    return tokenizer.apply_chat_template(case['prompt_messages'], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)

def generate(model, backbone, tokenizer, cases, torch):
    prompts = [render(tokenizer, c, flat=True) for c in cases]
    legacy_prompts = [render(tokenizer, c, flat=False) for c in cases]
    encoded = tokenizer(prompts, return_tensors='pt', add_special_tokens=False,
                        padding=True, truncation=False)
    lengths = encoded['attention_mask'].sum(1).tolist()
    width = encoded['input_ids'].shape[1]
    require(max(lengths) <= 8192, 'original BFCL prompt cap')
    encoded = {k: v.to(backbone.embed_tokens.weight.device) for k, v in encoded.items()}
    stops = eos_ids(tokenizer, model)
    kwargs = dict(max_new_tokens=2048, do_sample=False, pad_token_id=tokenizer.pad_token_id,
                  eos_token_id=sorted(stops), use_cache=True)
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**encoded, **kwargs)
    sync_all(torch)
    elapsed = time.monotonic() - started
    rows = []
    for i, case in enumerate(cases):
        ids = output[i, width:].tolist()
        stop = next((j for j, token in enumerate(ids) if token in stops), None)
        if stop is not None:
            ids = ids[:stop + 1]
        rows.append(dict(
            case=case, generation=dict(
                raw_response=tokenizer.decode(ids, skip_special_tokens=False),
                generated_token_ids=ids, n_generated_tokens=len(ids), n_prompt_tokens=int(lengths[i]),
                rendered_prompt_sha256=hashlib.sha256(prompts[i].encode()).hexdigest(),
                legacy_rendered_prompt_sha256=hashlib.sha256(legacy_prompts[i].encode()).hexdigest(),
                # Keep rendered text so interface differences remain independently reviewable.
                rendered_prompt=prompts[i],
                truncated=stop is None and len(ids) >= 2048,
                stop_reason='eos' if stop is not None else 'length' if len(ids) >= 2048 else 'other',
                requested_decoding={'do_sample': False, 'max_new_tokens': 2048},
                applied_generate_kwargs=kwargs,
                padding_side=tokenizer.padding_side, attention_mask_passed=True,
            ),
        ))
    return rows, elapsed
