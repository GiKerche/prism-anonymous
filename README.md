# PRISM

**Transfer and Limits of Activation Interventions in Tool-Calling Language Models**

This repository accompanies the anonymous review version of the final manuscript.
It contains the paper, experimental inputs, frozen fitting artifacts, response-level
outcomes, analysis code, and the three LoRA adapters evaluated in the paper.

[Read the paper](paper/main.pdf) · [Reproduce the results](REPRODUCIBILITY.md) ·
[Experimental code](experiments/README.md) · [Verification](reproduction/verification.json)

## Interactive explorer

The browser-only explorer in [`docs/`](docs/) contains 66,304 recorded LUMEN
responses. Compare matched baseline/intervention responses, inspect parsed tool
calls, select a model or prompt language, explore the 300 language-transfer cells,
and move through the measured Validation II dose grid. Native, shadow,
transported and receiver-native interventions remain distinct. The viewer does
not run a model, generate new outcomes, or execute tool calls.

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8765` in a current browser. GitHub Pages serves the same
files from the `gh-pages` branch. See [`site/README.md`](site/README.md) for
data provenance, integrity checks, and rebuilding the browser catalog.

## Reproduce the published results

The statistical checks run locally on CPU, without model downloads, API keys,
or calls to an external judge:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make reproduce
```

The anonymous host limits individual files to 8 MB. Eight larger artifacts
are stored as compressed parts below that limit. `make reproduce` restores
them automatically and checks their original SHA-256 hashes. To run individual
scripts, first run `python reproduction/restore_artifacts.py`.

For a smaller first check after restoration, run `python paper/analysis/reproduce_primary.py`.
To refit every construction probe as well, run `make refit`.

The checks cover the main effects, language transfer and Arabic, BFCL capability,
source and receiver controls, LoRA, SODA, layer selection, intervention directions,
PCA/ridge transport, dose selection, label scarcity, and the complete Llama70B
evaluator panel. Results are conditional on the recorded automatic labels;
reproducing the statistics does not independently validate the judges.

## Contents

| Location | Contents |
| --- | --- |
| `paper/` | Anonymous PDF, LaTeX/Overleaf sources, tables and figures |
| `paper/task/` | LUMEN scenarios, translations, tools, prompt contexts, family partitions and judge rubrics |
| `paper/data/` | Published numerical exports and compressed response-level endpoints |
| `paper/analysis/` | Family-bootstrap estimators and result verification scripts |
| `reproduction/` | Construction/validation activations, labels, probes, directions, benign anchors, maps and LoRA adapters |
| `experiments/` | Model revisions, generation/steering code, parsers, benchmark interfaces and LoRA training code |
| `docs/` | Interactive static explorer, anonymous paper and recorded response shards |
| `site/` | Offline browser-catalog builder and response/endpoint integrity verification |

This release contains one final study configuration. Both Llama models use
Prompt/Python for BFCL; Qwen and Ministral use their native tool interfaces.
Llama70B uses transported doses `+4/-4` and receiver-native doses `+8/-4`.
The Validation II capability tolerance is five percentage points.

The numerical artifacts make this repository larger than a paper-only release.
They contain numeric arrays, not base-model checkpoints. Base models and
external benchmark software retain their upstream access and licensing terms.

## Anonymous repository and website

The full reproducibility artifact is the **`main` branch at the repository
root**. An anonymous repository mirror must use that branch and include
`paper/`, `experiments/`, `reproduction/`, and the root documentation.
Enable downloads so readers can restore the numerical artifacts locally.

The **`gh-pages` branch contains only the interactive website**. Use it for a
separate website mirror. It does not contain the experimental code or fitting
artifacts and must not be used as the full reproducibility repository.

## Build the manuscript

```bash
make figures
make paper
```

Use `paper/main.tex` as the root file in Overleaf. The build needs a LaTeX
installation with the packages declared there; shell escape is disabled.

Private credentials, machine addresses, cluster configuration, scheduler jobs,
operational logs and local account paths are not part of this release.
Only aggregate scientific measurements of resource use are retained.
