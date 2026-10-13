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




## Command-Line Interface (CLI)

The project provides a Typer-based CLI for executing and managing
machine-learning workflows.

### Installation

From the project root:

```powershell
python -m pip install -e ".[dev]"
```

### Execution methods

The CLI can be invoked through either the installed executable:

```powershell
finance-ml --help
finance-ml check
finance-ml train --dry-run
```

Or directly through Python's module execution mechanism:

```powershell
python -m finance_ml --help
python -m finance_ml check
python -m finance_ml train --dry-run
```

Both methods invoke the same CLI application.

The installed `finance-ml` command is registered through
`[project.scripts]` in `pyproject.toml`, whereas `python -m finance_ml`
uses `finance_ml/__main__.py`.

### Training dry run

```powershell
finance-ml train --dry-run
```

This validates and displays training configuration without querying
Databricks, initializing TensorFlow, training a model, or writing
artifacts.

**Note:** Actual training execution is not connected to the CLI yet.
```
