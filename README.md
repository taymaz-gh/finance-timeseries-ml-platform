# Financial Time-Series ML Platform

An end-to-end project, built to practice production ML workflows, including data engineering, machine learning, MLOps, API deployment, orchestration, and business intelligence using simulated financial time-series data.

## Implemented holdout evaluation

The platform generates synthetic financial account trajectories, processes data through Databricks Bronze/Silver/Gold layers, and evaluates a version-pinned MLflow/Unity Catalog LSTM risk classifier with its saved preprocessing metadata.

Three **post-development holdout** cohorts (distinct from the original train/validation/test split) test different account/date combinations:

| Cohort | Accounts | Dates | Observations | Accuracy | Macro-F1 |
|---|---|---|---:|---:|---:|
| **Date Holdout** | Known | Future | 120,000 | 0.8988 | 0.8615 |
| **Account Holdout** | New | Historical | 60,000 | 0.8991 | 0.8595 |
| **Joint Holdout** | New | Future | 60,000 | 0.8967 | 0.8569 |

These are **observed results on synthetic data** for `finance_ml.models.lstm_risk_classifier` version 1; they are not estimates of production banking performance.

See [holdout pipeline documentation](docs/holdout_data_pipeline.md) for the two-dimensional account–date design, data flow diagrams, class-wise F1, confusion matrices, and limitations. See [model inference documentation](docs/model_inference_pipeline.md) for the pinned-model contract and batch-scoring interface. The end-to-end Databricks entry point is [`notebooks/10_build_future.ipynb`](notebooks/10_build_future.ipynb).
