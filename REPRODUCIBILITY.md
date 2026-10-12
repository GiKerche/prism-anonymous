# Reproducibility

## Scope

The deposited results correspond to the anonymous review manuscript. All default checks are offline CPU computations. They recompute the
reported statistics from recorded binary outcomes and reconstruct fitting and
selection computations from the deposited numerical inputs.

New generation or judging is a separate replication: stochastic sampling,
hardware, library implementations, and hosted judge availability can affect its
outputs. The release does not claim bit-for-bit equivalence of fresh API calls.
No new model generation or paid judging was used to prepare this repository.

The CPU reference environment is Python 3.13 with the versions in
`requirements.txt`. Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` for
the fitting checks. `make reproduce` does this automatically. Numerical
comparisons use explicit tolerances; scientific inputs are never modified by
the checking scripts.

## Verification commands

| Command | Reproduces |
| --- | --- |
| `python paper/analysis/reproduce_primary.py` | 24 main paired contrasts, missingness bounds, adjusted intervals and primary BFCL results |
| `python paper/analysis/reproduce_all_endpoints.py` | All exported LUMEN rates and contrasts, including language cells and Arabic, plus BFCL outcomes |
| `python paper/analysis/reproduce_additional.py` | Source/receiver controls and capability, and the 6,400 SODA trajectories |
| `python paper/analysis/reproduce_lora.py` | Per-replica rates, averaged paired effects, uncertainty and BFCL capability |
| `python reproduction/reproduce_selection.py` | Frozen probe predictions, full-depth layer selection, native/shadow directions and common activation scales |
| `python reproduction/reproduce_selection.py --refit-probes` | Additionally refits every construction-only logistic regression |
| `python reproduction/reproduce_maps.py` | All 60 PCA/ridge candidates, three selected maps, transported directions and benign diagnostic errors |
| `python reproduction/reproduce_doses.py` | Admissibility and the one-standard-error rule for all 24 selected signed doses, using the complete saved grid summaries |
| `python reproduction/reproduce_diagnostics.py` | Audited construction yield, 1,000 paired family subsamples per budget and language-vector geometry |
| `python reproduction/reproduce_judge_panel.py` | Complete Llama70B sensitivity analysis from 3,200 responses and 952 candidate responses |
| `python reproduction/reproduce_sensitivity.py` | Fixed 753-response audit, weighted evaluator sensitivity, language summaries and exploratory equivalence intervals |

`make figures` regenerates the primary benchmark figure, dose curves, scarcity,
language geometry, language matrices and layer-selection plot. Tables and
figures use the numerical exports in `paper/data/`; the analysis scripts check
those exports against the supplied endpoints.

## Data and estimator conventions

- Construction, Validation I, Validation II and test contain 35, 16, 17 and 40
  scenarios. A scenario family, including all translations and seeds, belongs
  to one partition. Historical screening is not a claim of a wholly unseen corpus.
- Native and shadow directions share BF16 activations but use separate BF16
  and NF4 response labels. A four-response fraction requires all four labels.
  Probes exclude missing binary labels individually.
- Native/shadow directions are estimated from within-language centered
  fractions, then pooled and normalized. No test response enters fitting.
- `reproduction/*/labels.json` contains the labels used for the frozen fits.
  `audited_label_changes.json` records later diagnostic corrections. The
  tested directions, probes and LoRA targets are not silently replaced by
  fits using the revised labels.
- The final endpoints retain missing labels as `null`. Pairing uses scenario,
  language, context and seed, with bootstrap resampling of entire families.
  Zero-dose aliases reuse their baseline observations.
- The Llama70B primary dose is `+4`, and its independently selected native
  comparator is `+8`. Same-dose fitted controls are retained, including the
  permuted-label control whose induction exceeds transported induction.
- SODA uses scenario-cluster resampling and executed trajectories; LUMEN and
  BFCL record proposed calls. These endpoints are distinct.

## Numerical artifacts

NPZ files contain only finite numeric arrays and are loaded with
`allow_pickle=False`. Construction and validation activations are stored by
layer with explicit scenario/language order. Anchor arrays are stored in
chunks of 256 examples, preserving train/validation/diagnostic order.
Each model directory contains its frozen probe coefficients and unit directions.

LoRA parameters are stored by transformer layer in
`reproduction/lora/adapter_<seed>/`; the original 55 construction targets and
training recipe accompany them. `experiments/restore_lora.py` reconstructs a
standard safetensors adapter. Base-model weights are obtained separately at
the exact revisions in `experiments/models.json`.

Scientific content hashes from the original frozen artifacts are retained
where applicable. Private file locations and operational receipts were removed.
`SHA256SUMS` covers the actual public files, so those hashes, rather than an
original private container hash, verify the distributed artifacts.

## Fresh experiments

See `experiments/README.md` for the preserved model loader, generation hooks,
parsers, benchmark interfaces, exact model revisions and LoRA training command.
GPU and external-judge computations are not run by `make reproduce`.
The exported GPU entry points were checked with their CPU dry-run modes;
they were not rerun on GPUs during publication preparation.

SODA's raw input texts and simulated states are obtained from its pinned
external dataset because they contain credential-shaped benchmark fixtures.
The repository supplies the exact 400-task manifest, dataset/input hashes,
restoration code, and all trajectory outcomes needed for statistical checks.
