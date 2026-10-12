# Experimental code

The offline reproduction in the repository root is sufficient to verify the
published measurements. This directory additionally provides the scientific
runtime functions used for generation, parsing, steering, judging requests,
and LoRA training, with portable entry points. Private storage, scheduler,
credential-routing and accounting code is omitted.

## Models and generation

`models.json` pins all six checkpoint revisions. Obtain those checkpoints
from their upstream repositories, under their access terms, and supply a
local checkpoint directory. The loader uses local files only and does not
read or include a model-download credential. Model class, BF16 precision,
SDPA attention, padding and chat-template behavior follow the recorded runtime.
The Llama70B loader retains the two-GPU placement used in the experiment.

Inspect the work before generation:

```bash
python experiments/run_lumen.py --model source_m5 --baseline --dry-run
python experiments/run_lumen.py --model receiver_m5 --dry-run
python experiments/run_bfcl.py --model receiver_m5 --baseline --dry-run
python experiments/train_lora.py --seed 26082701 --dry-run
```

For a fresh GPU replication, install `requirements-gpu.txt` in a separate
environment and replace `--dry-run` with `--checkpoint /path/to/checkpoint`.
New outputs are written under the ignored `outputs/` directory. They are not
promoted into the deposited result tables. The recorded core GPU versions are
PyTorch 2.8.0 with CUDA 12.8 and Transformers 5.15.0.

`run_lumen.py` supports the four partitions and both prompt contexts.
The default final doses come from the frozen model configuration. Arabic is
restricted to final evaluation. An explicit `--dose` is accepted only on
Validation II, using the published grid. Construction and Validation I require
`--baseline`. `--direction shadow_pooled`, `native_pt`, or another deposited
direction selects that unit vector. A zero dose shares the baseline output
location. No base-model weights are updated by steering.

LUMEN uses temperature 0.7, top-p 0.95 and 1,024 output tokens. BFCL and SODA
use greedy decoding with 2,048 output tokens. Qwen thinking is disabled as in
this paper. Batch size, tokenization and hardware can change fresh stochastic
outputs; the response-level data in `paper/data/` remain the reference for the
published statistics. The deposited construction labels include both BF16 and
NF4 supervision; the portable final-intervention runner loads BF16 models.

## BFCL and SODA

`run_bfcl.py` uses Prompt/Python for both Llama models and native tools for Qwen
and Ministral. Native model inputs were exported from the actual recorded
generation batches. BFCL generation inputs do not contain expected answers;
the separate `bfcl_*_gold.json` files are for the official checker only.
The 86 Validation II and 914 test cases retain their original membership.

`benchmarks.json` identifies the external code and data snapshots. The BFCL
AST checker is supplied by its upstream repository, rather than bundled with
provider SDKs. `bfcl_checker.py` adapts the official checker to the stored case
format. It keeps whole-response malformed-call handling separate from the
LUMEN/SODA action parser.

After generating a BFCL arm, score it against the pinned upstream checker:

```bash
python experiments/score_bfcl.py --input outputs/bfcl \
  --benchmark-root external/bfcl/berkeley-function-call-leaderboard
```

The scorer verifies the checker files against `bfcl_checker_hashes.json` and
reports whether every reserved case is present. Prompt/Python uses the original
prompting AST decoder, with expression evaluation restricted to `ast.literal_eval`.

`soda_runtime.py` preserves the executed-trajectory loop, including model
feedback, complete-call parsing and the five-round limit. The 400 task identities
and exact decoding recipe are in `inputs/soda_*`. SODA environment code is an
external dependency at the recorded revision; it is not replaced by real
services. `run_soda.py` verifies that supplied environment files match the
recorded snapshot before execution.

`inputs/soda_tasks.json` contains task identities, not the raw prompts and
simulated states, which contain credential-shaped benchmark fixtures. Fresh execution
requires `--dataset` pointing to the full-interaction test Parquet from the
dataset revision in `benchmarks.json` (install `pyarrow` to read it).
`soda_inputs.py` reconstructs all 400 inputs in memory and verifies the exact
dataset and complete input hashes. The runner refuses to execute the manifest
in place of the original inputs. These omissions do not affect the deposited
trajectory labels or any offline statistical check.

## Labels and audits

`prism/semantic_judging.py` and `prism/soda_semantic_judging.py` preserve request
construction, rubrics, response schemas and decision validation. They contain
no provider client or credentials. Construction/layer-validation resolution
requires two matching valid votes. Final primary labels and additional audit
votes remain separate. Missing decisions stay missing.

Regenerating labels requires independent access to the dated judges stated in
the manuscript. The offline checks use the deposited labels, without querying
an API or claiming that hosted judges are reproducible bit for bit.

## LoRA

`train_lora.py` trains a fresh replica using the original 55 positive NF4
construction targets, rank 16, alpha 32, dropout 0.05, five epochs and 20
optimizer steps. The final partial batch is preserved; loss is masked on the
prompt and weighted by completion-token count within each accumulation group.
No validation or test example enters this training set.

The three evaluated adapters are already deposited as numeric arrays. To use
one without retraining:

```bash
python experiments/restore_lora.py --seed 26082701 --output outputs/adapter_26082701.safetensors
```

They are loaded through `prism.lora.inject_all_backbone_linears` and
`load_adapter_state_dict` on the pinned Qwen3.5-9B checkpoint. The release does
not contain a base-model checkpoint, pickle checkpoints, or optimizer states.
