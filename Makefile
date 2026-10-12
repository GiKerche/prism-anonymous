PYTHON ?= python
export OPENBLAS_NUM_THREADS = 1
export OMP_NUM_THREADS = 1

.PHONY: reproduce refit figures paper verify

reproduce:
	$(PYTHON) paper/analysis/reproduce_all_endpoints.py
	$(PYTHON) paper/analysis/reproduce_additional.py
	$(PYTHON) paper/analysis/reproduce_lora.py
	$(PYTHON) reproduction/reproduce_selection.py
	$(PYTHON) reproduction/reproduce_maps.py
	$(PYTHON) reproduction/reproduce_doses.py
	$(PYTHON) reproduction/reproduce_diagnostics.py
	$(PYTHON) reproduction/reproduce_judge_panel.py
	$(PYTHON) reproduction/reproduce_sensitivity.py

refit:
	$(PYTHON) reproduction/reproduce_selection.py --refit-probes

figures:
	cd paper && $(PYTHON) make_reviewer_figures.py
	cd paper && $(PYTHON) make_language_figures.py
	cd paper && $(PYTHON) make_probe_selection_figure.py

paper:
	cd paper && pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
	cd paper && bibtex main
	cd paper && pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex

verify:
	sha256sum --check SHA256SUMS
