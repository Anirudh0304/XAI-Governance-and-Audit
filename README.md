# explainable-ai-finance

Minimal project scaffold for explainable AI in finance.

Structure created for data, src, reports, dashboards, and notebooks.

Usage:
- Populate `data/raw/credit.csv` with the original dataset.
- Run training in `src/model_training.py` to produce `artifacts/model_xgb.json`.
- Use `src/explainability_layer.py` and `src/metrics/*` for explainability and fairness evaluation.
