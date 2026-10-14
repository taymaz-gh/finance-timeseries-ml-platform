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

The platform provides a Typer-based CLI for model training,
synthetic-data simulation, and account/date-based inference.
It supports standalone execution with local Spark and
TensorFlow/Keras, together with optional Databricks and
Unity Catalog integration.

### Installation

From the repository root:

```powershell
python -m pip install -e ".[dev]"
```

Verify the installation:

```powershell
finance-ml check
finance-ml --help
```

The CLI can also be invoked using `python -m finance_ml`.

### Supported workflows

| Workflow | Data source | Model source or destination | Execution |
|---|---|---|---|
| Local training | Synthetic data | Saved local Keras bundle | Local |
| Databricks-backed training | Databricks Gold tables | Local bundle; optional MLflow logging and UC registration | Local |
| Local inference | Replayed synthetic observations | Saved local Keras bundle | Local |
| Databricks/UC inference | Persisted Databricks Gold tables | Registered Unity Catalog model | Local |

Training and inference are separate operations. Inference never
re-trains models, refits preprocessing statistics, or registers
models in Unity Catalog.

### Training

Local synthetic training:

```powershell
finance-ml train `
    --data-source synthetic `
    --config configs/training_synthetic.yaml `
    --synthetic-accounts 20 `
    --synthetic-days 60 `
    --synthetic-seed 42 `
    --dry-run
```

Remove `--dry-run` to execute the training run.


The synthetic training workflow applies controlled Bronze data-quality
injection before Silver cleaning and Gold feature engineering.
With the `reference_v1` schema profile, the LSTM receives 41
ordered features: 22 numerical features, 16 missingness
indicators, and 3 categorical one-hot features. Imputation
and scaling are fitted using the training partition only.


Databricks-backed training configuration check:

```powershell
finance-ml train `
    --data-source databricks `
    --query-method connector `
    --run-mode development `
    --dry-run
```

For Databricks-backed training, the standalone CLI retrieves
Gold data through the Databricks SQL Connector or SQLAlchemy,
then performs Spark preprocessing and model training locally.

MLflow logging and Unity Catalog registration are controlled
by explicit configuration. UC registration additionally
requires `--allow-uc-registration` during execution.

### Inference

Inference selects account identifiers and an inclusive
calendar interval. The selected model's persisted preprocessing
metadata defines the exact input feature schema.

**Local model inference:**

```powershell
$bundle = "artifacts\synthetic_training\<your-run-directory>"
```

Replace <your-run-directory> with the run directory printed after training. Local model artifacts are not included in the Git repository.

```powershell
finance-ml infer `
    --data-source synthetic `
    --model-source local `
    --bundle "$bundle" `
    --account-id 21 `
    --start-date 2026-02-10 `
    --end-date 2026-02-12 `
    --smoke
```

Local synthetic replay requires a checkpoint-bound provenance
profile. For older bundles, this can be created using
`finance-ml profile-init` with the verified training parameters.

**Verified local model:** As a small-scale demonstration, A seven-timestep LSTM with 41
features, trained using 20 synthetic development accounts and five epochs.
The selected local bundle was evaluated on 10 previously unseen
accounts (21–30), covering January 7–March 1, 2026.

| Metric | Account Holdout result |
|---|---:|
| Predictions | 540/540 |
| Accuracy | 0.7815 |
| Macro-F1 | 0.6956 |

These are synthetic-data results, not estimates of real-world
financial-risk performance. Local inference replays clean synthetic
observations and applies the model's saved preprocessing without
refitting. The model's training data included controlled Bronze
data-quality corruption.

The local model is separate from the registered Unity Catalog
model v1. Both have 41 input features, but different trained
weights, fitted preprocessing parameters, and development populations.


**Registered Unity Catalog model inference:**

```powershell
finance-ml infer `
    --data-source databricks `
    --model-source uc `
    --model-name finance_ml.models.lstm_risk_classifier `
    --model-version 1 `
    --account-id 521 `
    --start-date 2026-03-10 `
    --end-date 2026-03-12 `
    --smoke `
    --evaluate
```

The Databricks/UC inference workflow:

1. Retrieves selected Gold features and required historical context.
2. Loads the specified registered model and its verified preprocessing metadata.
3. Applies the saved transformations and executes inference locally.
4. Optionally retrieves ground-truth labels for evaluation.
5. Saves predictions, coverage, and evaluation reports locally.

No Databricks tables or registered models are modified.

### Execution modes and artifacts

- `--dry-run`: validates configuration and displays an execution plan without training or inference.
- `--smoke`: executes a small, real inference request to verify the end-to-end workflow.
- Normal execution: processes the requested observations subject to resource and validation limits.
- `--evaluate`: enables ground-truth evaluation for Databricks-backed inference.

Inference results are written to run-specific directories under
`artifacts/inference/`, including:

- `predictions.csv`
- `coverage.csv`
- `inference_report.json`

Evaluation reports include accuracy, fixed-four-class macro-F1,
per-class metrics, and confusion matrices when labels are available.

### Documentation

- [Complete CLI guide](docs/cli_guide.md) — commands, configurations, execution modes, PowerShell examples, and architecture diagrams.
- [Model training pipeline](docs/model_training_pipeline.md) — training and MLflow integration.
- [Model inference pipeline](docs/model_inference_pipeline.md) — model loading, preprocessing contracts, and batch inference.
- [Holdout evaluation](docs/holdout_data_pipeline.md) — Date, Account, and Joint Holdout cohorts.
- [Modeling and metric conventions](docs/modeling_rules.md) — sequence modeling and evaluation methodology.