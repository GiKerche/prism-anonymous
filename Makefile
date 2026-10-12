PYTHON ?= python
.DEFAULT_GOAL := reproduce
export OPENBLAS_NUM_THREADS = 1
export OMP_NUM_THREADS = 1

.PHONY: restore reproduce refit figures paper verify

restore:
	$(PYTHON) reproduction/restore_artifacts.py

reproduce: restore
	$(PYTHON) paper/analysis/reproduce_all_endpoints.py
	$(PYTHON) paper/analysis/reproduce_additional.py
	$(PYTHON) paper/analysis/reproduce_lora.py
	$(PYTHON) reproduction/reproduce_selection.py
	$(PYTHON) reproduction/reproduce_maps.py
	$(PYTHON) reproduction/reproduce_doses.py
	$(PYTHON) reproduction/reproduce_diagnostics.py
	$(PYTHON) reproduction/reproduce_judge_panel.py
	$(PYTHON) reproduction/reproduce_sensitivity.py

refit: restore
	$(PYTHON) reproduction/reproduce_selection.py --refit-probes

figures: restore
	cd paper && $(PYTHON) make_reviewer_figures.py
	cd paper && $(PYTHON) make_language_figures.py
	cd paper && $(PYTHON) make_probe_selection_figure.py

paper: restore
	cd paper && pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
	cd paper && bibtex main
	cd paper && pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex

verify: restore
	sha256sum --check SHA256SUMS
