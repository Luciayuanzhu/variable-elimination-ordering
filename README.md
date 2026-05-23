# Variable Elimination Ordering Experiments

This repository implements an experiment pipeline for comparing variable
elimination ordering heuristics for exact inference in probabilistic graphical
models.

The compared methods are:

- min-degree
- min-fill
- weighted min-fill
- hybrid weighted min-fill plus downstream clique-pressure heuristic
- bounded-lookahead oracle on small pilot graphs

The experiments generate five graph topology families: low-treewidth
chains/trees/polytrees, synthetic random graphs, grids, scale-free/hub graphs,
and moralized Bayesian-network-style benchmarks.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiments.py
python scripts/make_plots.py
python scripts/write_report.py
latexmk -pdf -interaction=nonstopmode -halt-on-error report/results_report.tex
```

For the expanded v2 experiment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiments_v2.py
python scripts/make_plots_v2.py
python scripts/write_report_v2.py
latexmk -pdf -interaction=nonstopmode -halt-on-error report/results_report_v2.tex
```

Generated artifacts:

- `results/results.csv`
- `figures/*.pdf`
- `report/results_report.pdf`
- `results/results_v2.csv`
- `report/results_report_v2.pdf`
