# Financial Time-Series ML Platform

An end-to-end project, built to practice production ML workflows, including data engineering, machine learning, MLOps, API deployment, orchestration, and business intelligence using simulated financial time-series data.

## Quick start: local CSV prediction dashboard

This section is for someone **cloning the repository onto a new Windows laptop**. The application accepts **raw, unlabeled account-day observations** and displays predictions from a **previously trained local Keras bundle**. The model bundle itself is **not included in Git**. No Databricks credentials are required for this local workflow.

### A. Technical setup (once per machine)

Prerequisites: Git, Python **3.12**, Java **17** (for local PySpark), and [uv](https://docs.astral.sh/uv/getting-started/installation/). Run the commands below in **PowerShell** from the repository root, after cloning:

```powershell
git clone https://github.com/taymaz-gh/finance-timeseries-ml-platform.git
cd finance-timeseries-ml-platform
uv sync --locked --extra api --extra dashboard --extra dev
Copy-Item .env.example .env
```

The `uv` command creates/synchronizes `.venv` and installs API, dashboard, and testing dependencies from the committed lockfile. If using an activated environment instead, the corresponding `python` commands below work normally. **Do not commit `.env` or generated `artifacts/`.**

### B. Obtain a local trained model

**Option 1 — Already have a model:** Copy the complete **local Keras bundle directory** from an existing training run (not just the `.keras` file) into a local folder such as `artifacts/synthetic_training/<run-directory>`. The bundle contains `best_model.keras`, `preprocessing_metadata.json`, and `bundle_manifest.json`; it must pass the backend's bundle-integrity checks. Model files are not distributed by the repository.

**Option 2 — Train a new model locally:** The existing Typer CLI supports synthetic-data training:

```powershell
uv run --locked --extra api --extra dashboard --extra dev finance-ml train `
    --data-source synthetic `
    --config configs/training_synthetic.yaml `
    --synthetic-accounts 20 `
    --synthetic-days 60 `
    --synthetic-seed 42
```

Training uses the local Spark/TensorFlow stack and may take some time. Inspect the CLI output for the resulting bundle path under `artifacts/synthetic_training/`. The optional `--dry-run` checks configuration **without creating a model**. See the existing [CLI guide](docs/cli_guide.md) for details; the stronger configuration `configs/training_synthetic_stronger.yaml` is another experiment, not a required setup step.

Edit `.env` and set the **directory** (not `best_model.keras`) using an absolute path, for example:

```dotenv
FINANCE_ML_API_BUNDLE_DIR=D:/Projects/finance-timeseries-ml-platform/artifacts/synthetic_training/<run-directory>
```

Replace `<run-directory>` with the actual folder name and use your own checkout location. Forward slashes work on Windows. Other fields in `.env.example` relate to optional Databricks workflows and can remain blank for local API inference.

### C. Generate an uploadable sample CSV

Run the **existing sample generator** from the repository root:

```powershell
uv run --locked --extra api --extra dashboard --extra dev python scripts/06_create_sample_upload_csv.py
```

It creates `artifacts/sample_accounts.csv` with **three synthetic accounts and 14 daily observations per account**, excluding the risk-state target labels. This is an example of the **user-upload format**, not a pretrained model or predictions.

The CSV requires **exactly these 12 column names** (order is flexible):

```text
account_id,date,account_type,balance,cash_inflow,cash_outflow,transaction_count,credit_utilization,amount_due,amount_paid,payment_ratio,days_past_due
```

Dates must be in `YYYY-MM-DD` format, numeric fields must be numeric or missing, and uploaded CSVs must not contain `risk_state` or `risk_state_name`. The API limits files to **10 MiB** and **20,000 rows**. Duplicate account/date records are checked, then handled by the existing cleaning pipeline. Inference requires sufficient sequential history for each account.

### D. Start both local services

Open **two terminals** at the repository root (use the project environment in each):

**Terminal 1 — FastAPI model-serving backend**

```powershell
uv run --locked --extra api --extra dashboard --extra dev python -m uvicorn finance_ml.api.main:app --host 127.0.0.1 --port 8000
```

Verify that `http://127.0.0.1:8000/ready` returns `{"status":"ready"}`. `http://127.0.0.1:8000/docs` exposes the automatically generated interactive API reference. An HTTP 503 from `/ready` means the saved model is not available; check the bundle path and backend logs.

**Terminal 2 — Streamlit user interface**

```powershell
uv run --locked --extra api --extra dashboard --extra dev python -m streamlit run src/finance_ml/dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Open **http://127.0.0.1:8501** in your browser. Both commands bind to the local machine only; this development setup is **not a public deployment**. Stop a server with `Ctrl+C` in its terminal.

### E. Non-technical user workflow

With both services already started by a technical operator, the user only needs the Streamlit page:

1. Open `http://127.0.0.1:8501` on that same computer.
2. Click **Browse files** and choose a raw unlabeled CSV, such as `artifacts/sample_accounts.csv`.
3. Click **Generate predictions**. Streamlit asks FastAPI to validate the CSV, then requests actual local-model inference.
4. Review the summary, select an account, inspect its predicted risk-state timeline, class-probability plot, distribution and coverage table.
5. Click **Download all predictions (CSV)** to export results.

The backend automatically saves full historical runs under `artifacts/api_inference/runs/<run-id>/` and refreshes `artifacts/api_inference/latest_predictions.csv`, `latest_coverage.csv`, `latest_account_context.csv`, and `latest_summary.json`. The dashboard also holds the current result in the user's Streamlit session; it does **not** train or reload the model on each click.

**Interpretation and safety:** These are **same-date risk classifications** based on current and earlier uploaded observations, **not forecasts of unobserved future data**. Synthetic training data do not validate real-world banking performance. The application currently has **no authentication or HTTPS**; keep it loopback-only and do not upload sensitive customer data to this development environment.

### F. Tests and troubleshooting

```powershell
uv run --locked --extra api --extra dashboard --extra dev python -m pytest -q tests/test_dashboard_app.py
```

For full verification, run `python -m pytest -q` in the synchronized environment. Common problems: `Connection refused` indicates FastAPI is not running; HTTP 503 means no loaded model; an HTTP 422 usually signals incorrect CSV columns, types, dates, or target leakage; too few observations produce `insufficient_context`. A dataset with fewer than the trained model's sequence length (usually 7) per account cannot produce predictions. Consult [API and dashboard developer guide](docs/api_dashboard_guide.md) for complete workflow diagrams and module ownership.

---

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

- [API and dashboard developer guide](docs/api_dashboard_guide.md) — endpoints, module responsibilities, architecture, and tests.

- [Complete CLI guide](docs/cli_guide.md) — commands, configurations, execution modes, PowerShell examples, and architecture diagrams.
- [Model training pipeline](docs/model_training_pipeline.md) — training and MLflow integration.
- [Model inference pipeline](docs/model_inference_pipeline.md) — model loading, preprocessing contracts, and batch inference.
- [Holdout evaluation](docs/holdout_data_pipeline.md) — Date, Account, and Joint Holdout cohorts.
- [Modeling and metric conventions](docs/modeling_rules.md) — sequence modeling and evaluation methodology.