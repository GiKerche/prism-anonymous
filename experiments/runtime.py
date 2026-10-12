"""Model loading, rendering, generation and steering used in the final experiments.

Scientific functions retained; private staging and retired selection guards omitted.
"""
from __future__ import annotations
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable

LANGUAGES=("pt","en","es","zh","hi")

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)[:-1]).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def require_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise RuntimeError(
            f"{label} differs: observed={observed!r} expected={expected!r}"
        )

def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"immutable output differs: {path}")
        return
    partial = path.with_name(path.name + f".partial.{os.getpid()}")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if partial.exists():
            partial.unlink()

def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, canonical_bytes(value))

def model_spec(contract: dict[str, Any], slot: str, role: str) -> dict[str, Any]:
    validate_contract(contract)
    require(slot in ("M1", "M3", "M5"), "unregistered model slot")
    require(role in ("source", "receiver"), "unregistered model role")
    value = contract["models"][slot][role]
    require(isinstance(value, dict), "model specification is not an object")
    return value

def sync_all(torch: Any) -> None:
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)

def _loading_errors(info: dict[str, Any], allowed_unexpected: tuple[str, ...]) -> dict[str, Any]:
    missing = list(info.get("missing_keys", []))
    unexpected = list(info.get("unexpected_keys", []))
    mismatched = list(info.get("mismatched_keys", []))
    errors = list(info.get("error_msgs", []))
    bad_unexpected = [key for key in unexpected if not key.startswith(allowed_unexpected)]
    return {
        "missing": missing,
        "bad_unexpected": bad_unexpected,
        "mismatched": mismatched,
        "errors": errors,
        "allowed_unexpected_count": len(unexpected) - len(bad_unexpected),
    }

def load_model(
    *, contract: dict[str, Any], slot: str, role: str, model_root: Path, torch: Any
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load one pinned text model and return model, tokenizer, text backbone, audit."""

    import transformers
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        LlamaForCausalLM,
        Ministral3ForCausalLM,
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
    )

    spec = model_spec(contract, slot, role)
    require(model_root.is_dir() and not model_root.is_symlink(), "materialized model root is unsafe")
    full_config = AutoConfig.from_pretrained(
        model_root, local_files_only=True, token=False, trust_remote_code=False
    )
    key_mapping: dict[str, str] | None = None
    allowed_unexpected: tuple[str, ...] = ()
    device_map: dict[str, int] | None = None
    if slot == "M1" and role == "source":
        loader_config = full_config
        architecture_config = full_config
        loader = Qwen3_5ForCausalLM
        backbone_path = "model"
    elif slot == "M1" and role == "receiver":
        loader_config = full_config
        architecture_config = full_config.text_config
        loader = Qwen3_5ForConditionalGeneration
        backbone_path = "model.language_model"
    elif slot == "M3":
        loader_config = full_config.text_config
        architecture_config = loader_config
        loader = Ministral3ForCausalLM
        backbone_path = "model"
        index_path = model_root / "model.safetensors.index.json"
        require(index_path.is_file(), "Ministral index is absent")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        require(isinstance(index, dict), "Ministral index is not an object")
        weight_map = index.get("weight_map")
        require(isinstance(weight_map, dict), "Ministral weight map is absent")
        key_mapping = {
            key: key.removeprefix("language_model.")
            for key in weight_map
            if key.startswith("language_model.")
        }
        require(len(key_mapping) > 0, "Ministral text key map is empty")
        allowed_unexpected = ("vision_tower.", "multi_modal_projector.")
    elif slot == "M5":
        loader_config = full_config
        architecture_config = full_config
        loader = LlamaForCausalLM
        backbone_path = "model"
        if role == "receiver":
            require(torch.cuda.device_count() == 2, "M5 receiver requires exactly two GPUs")
            device_map = {"model.embed_tokens": 0, "model.rotary_emb": 0}
            device_map.update(
                {
                    f"model.layers.{layer}": 0 if layer < 40 else 1
                    for layer in range(80)
                }
            )
            device_map["model.norm"] = 1
            device_map["lm_head"] = 1
    else:
        raise RuntimeError("unregistered model loader")

    require_equal(
        int(architecture_config.num_hidden_layers), int(spec["blocks"]), "model depth"
    )
    require_equal(
        int(architecture_config.hidden_size), int(spec["hidden_size"]), "hidden size"
    )
    tokenizer_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "token": False,
        "trust_remote_code": False,
    }
    if slot == "M3":
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_root, **tokenizer_kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "config": loader_config,
        "local_files_only": True,
        "token": False,
        "trust_remote_code": False,
        "use_safetensors": True,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
        "output_loading_info": True,
    }
    if key_mapping is not None:
        kwargs["key_mapping"] = key_mapping
    if device_map is not None:
        kwargs["device_map"] = device_map
    print(f"MAIN_LAYER_MODEL_LOAD_START slot={slot} role={role}", flush=True)
    started = time.monotonic()
    model, loading_info = loader.from_pretrained(model_root, **kwargs)
    load_seconds = time.monotonic() - started
    errors = _loading_errors(loading_info, allowed_unexpected)
    require(
        not errors["missing"]
        and not errors["bad_unexpected"]
        and not errors["mismatched"]
        and not errors["errors"],
        f"checkpoint load is incomplete: {errors}",
    )
    if device_map is None:
        require_equal({str(parameter.device) for parameter in model.parameters()}, {"cpu"}, "CPU load")
        model.to(torch.device("cuda:0"))
    model.eval()
    sync_all(torch)
    require(not any(parameter.is_meta for parameter in model.parameters()), "model retains meta parameters")
    backbone: Any = model
    for component in backbone_path.split("."):
        backbone = getattr(backbone, component)
    require_equal(len(backbone.layers), int(spec["blocks"]), "materialized text layers")
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    expected_devices = ["cuda:0", "cuda:1"] if device_map is not None else ["cuda:0"]
    require_equal(devices, expected_devices, "parameter devices")
    floating_dtypes = sorted(
        {str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()}
    )
    require_equal(floating_dtypes, ["torch.bfloat16"], "floating parameter dtypes")
    audit = {
        "slot": slot,
        "role": role,
        "model_id": spec["id"],
        "model_revision": spec["revision"],
        "loader_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "backbone_path": backbone_path,
        "blocks": len(backbone.layers),
        "hidden_size": int(architecture_config.hidden_size),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "devices": devices,
        "load_seconds": load_seconds,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "allowed_unexpected_checkpoint_keys": errors["allowed_unexpected_count"],
        "cpu_or_disk_offload": False,
    }
    print(
        f"MAIN_LAYER_MODEL_LOAD_PASS slot={slot} role={role} seconds={load_seconds:.2f}",
        flush=True,
    )
    return model, tokenizer, backbone, audit

def render_case(tokenizer: Any, case: dict[str, Any], tools: list[dict[str, Any]]) -> str:
    kwargs: dict[str, Any] = {
        "tools": tools,
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    messages = [
        {"role": "system", "content": case["system_prompt"]},
        {"role": "user", "content": case["user_prompt"]},
    ]
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    require(isinstance(rendered, str) and bool(rendered), "chat template emitted empty text")
    return rendered

def render_anchor(tokenizer: Any, prompt: str) -> str:
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    require(isinstance(rendered, str) and bool(rendered), "anchor template emitted empty text")
    return rendered

def eos_ids(tokenizer: Any, model: Any) -> set[int]:
    values: list[Any] = [
        tokenizer.eos_token_id,
        getattr(model.generation_config, "eos_token_id", None),
    ]
    result: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, int):
            result.add(value)
        else:
            result.update(int(item) for item in value)
    require(bool(result), "model has no EOS token")
    return result

def input_device(backbone: Any) -> Any:
    return backbone.embed_tokens.weight.device

def generate_batch(
    *,
    model: Any,
    backbone: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    tools_by_language: dict[str, list[dict[str, Any]]],
    seed: int,
    torch: Any,
    max_new_tokens: int = 1024,
) -> tuple[list[dict[str, Any]], float]:
    from prism.action_labels import action_metrics
    from prism.degeneration import diagnose_degeneration
    from prism.tool_parser import parse_tool_calls_detailed, strip_tool_call_blocks

    rendered = [
        render_case(tokenizer, case, tools_by_language[case["language"]])
        for case in cases
    ]
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
        truncation=False,
    )
    prompt_tokens = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
    require(max(prompt_tokens) <= 8192, "causal prompt exceeds token cap")
    encoded_width = int(encoded["input_ids"].shape[-1])
    device = input_device(backbone)
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    stops = eos_ids(tokenizer, model)
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=sorted(stops),
            use_cache=True,
        )
    sync_all(torch)
    elapsed = time.monotonic() - started
    rows: list[dict[str, Any]] = []
    for position, case in enumerate(cases):
        padded = output[position, encoded_width:].tolist()
        stop = next((i for i, token in enumerate(padded) if token in stops), None)
        tokens = padded if stop is None else padded[: stop + 1]
        stopped = stop is not None
        truncated = not stopped and len(tokens) >= max_new_tokens
        raw = tokenizer.decode(tokens, skip_special_tokens=False)
        parsed = parse_tool_calls_detailed(raw)
        calls = parsed.calls
        tools = tools_by_language[case["language"]]
        allowed = {
            tool["function"]["name"]
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        }
        metrics = action_metrics(calls)
        require(metrics["L2"] <= metrics["L1"] <= metrics["L0"], "L0/L1/L2 nesting differs")
        rows.append(
            {
                "rendered_prompt_sha256": hashlib.sha256(rendered[position].encode()).hexdigest(),
                "raw_response": raw,
                "response_text": strip_tool_call_blocks(raw),
                "generated_token_ids": tokens,
                "n_prompt_tokens": prompt_tokens[position],
                "n_generated_tokens": len(tokens),
                "stop_reason": "eos" if stopped else "length" if truncated else "other",
                "truncated": truncated,
                "parse": parsed.to_dict(),
                "action_metrics": metrics,
                "recognized_tool_schema": bool(calls)
                and all(call.get("name") in allowed for call in calls),
                "malformed": parsed.status == "malformed",
                "degeneration": diagnose_degeneration(raw).to_dict(),
                "elapsed_seconds": elapsed,
                "elapsed_scope": "shared_batch",
            }
        )
    return rows, elapsed

class Intervention:
    def __init__(self, *, torch: Any, layer: Any, vector: Any, strength: float):
        self.torch = torch
        self.layer = layer
        self.vector = vector
        self.strength = float(strength)
        self.handle: Any = None
        self.calls = 0
        self.positions = 0

    def __enter__(self) -> "Intervention":
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            require(self.torch.is_tensor(hidden) and hidden.ndim == 3, "intervention shape")
            self.calls += 1
            self.positions += int(hidden.shape[0] * hidden.shape[1])
            shift = self.vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
            shifted = hidden + self.strength * shift
            return (shifted, *output[1:]) if isinstance(output, tuple) else shifted

        self.handle = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()

def intervention_context(
    *, torch: Any, layer: Any, vector: Any | None, strength: float
) -> Any:
    if vector is None:
        return nullcontext(None)
    return Intervention(torch=torch, layer=layer, vector=vector, strength=strength)

def load_vector(path: Path, hidden_size: int) -> Any:
    import numpy as np

    require(path.is_file() and not path.is_symlink(), f"vector is unsafe: {path}")
    require_equal(path.stat().st_size, hidden_size * 4, "vector bytes")
    value = np.fromfile(path, dtype="<f4")
    require_equal(value.shape, (hidden_size,), "vector shape")
    require(bool(np.isfinite(value).all()), "vector contains nonfinite values")
    norm = float(np.linalg.norm(value.astype(np.float64)))
    require(math.isfinite(norm) and abs(norm - 1.0) <= 2e-5, "vector is not unit normalized")
    return value

def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    require(size > 0, "chunk size must be positive")
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]
