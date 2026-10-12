"""Exact batching and completion-only masking used for the final LoRA replicas."""
from typing import Any
import random

def tokenize_examples(tokenizer: Any, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        prompt = row["rendered_prompt"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion = row["completion_token_ids"]
        if len(prompt_ids) != row["recorded_prompt_tokens"]:
            raise RuntimeError(
                f"prompt token count drifted for {row['scenario_id']}/{row['language']}/{row['source_seed']}"
            )
        if len(prompt_ids) + len(completion) > 8192:
            raise RuntimeError("tokenized LoRA example would be truncated")
        result.append({
            "split": row["split"],
            "scenario_id": row["scenario_id"],
            "language": row["language"],
            "source_seed": row["source_seed"],
            "input_ids": prompt_ids + completion,
            "prompt_length": len(prompt_ids),
            "completion_length": len(completion),
        })
    lengths = [len(row["input_ids"]) for row in result]
    return result, {"min_sequence_tokens": min(lengths), "max_sequence_tokens": max(lengths)}

def make_microbatches(indices: list[int], micro_batch_size: int) -> list[list[int]]:
    return [indices[offset : offset + micro_batch_size] for offset in range(0, len(indices), micro_batch_size)]

def collate(torch: Any, examples: list[dict[str, Any]], indices: list[int], pad_token_id: int) -> dict[str, Any]:
    selected = [examples[index] for index in indices]
    max_length = max(len(row["input_ids"]) for row in selected)
    input_ids = torch.full((len(selected), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(selected), max_length), dtype=torch.long)
    labels = torch.full((len(selected), max_length), -100, dtype=torch.long)
    completion_tokens = 0
    for index, row in enumerate(selected):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        length = values.numel()
        input_ids[index, :length] = values
        attention_mask[index, :length] = 1
        labels[index, row["prompt_length"] : length] = values[row["prompt_length"] :]
        completion_tokens += int(row["completion_length"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "completion_tokens": completion_tokens,
    }

def groups_for_epoch(
    count: int, *, seed: int, epoch: int, micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> list[list[list[int]]]:
    indices = list(range(count))
    random.Random(seed + epoch * 100003).shuffle(indices)
    microbatches = make_microbatches(indices, micro_batch_size)
    return [
        microbatches[offset : offset + gradient_accumulation_steps]
        for offset in range(0, len(microbatches), gradient_accumulation_steps)
    ]
