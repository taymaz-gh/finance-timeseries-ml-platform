# CLI guide — Financial Time-Series ML Platform

Use `finance-ml --help`, `finance-ml train --help`, `finance-ml infer --help`, and `finance-ml profile-init --help` for the authoritative installed option descriptions. The CLI separates **training**, **single-/multi-account inference**, **synthetic simulation preview**, and **provenance registration**. Ordinary inference is defined by **account identifiers and an inclusive calendar interval**, not by a holdout selector.

## 1. Commands and execution modes

| Command | Purpose | Dry run | Smoke/full execution |
|---|---|---|---|
| `finance-ml check` | Verify CLI installation | N/A | Local check |
| `finance-ml train` | Fit a new LSTM with frozen preprocessing artifacts | Yes | Actual training; bounded synthetic experiment available |
| `finance-ml profile-init` | Associate an existing Keras checkpoint with user-declared synthetic replay parameters | N/A | Writes **only** a JSON sidecar, never the model or UC tables |
| `finance-ml infer` | Score requested account/date observations with a saved bundle | Yes | `--smoke` validates a **small user request**; otherwise full bounded execution |
| `finance-ml simulate` | Preview synthetic source streams without training or persistence | N/A | In-memory simulation preview |

**Smoke execution is not a dry run:** it initializes real Spark/Keras. It does **not** silently substitute different accounts/dates; requests for more than two accounts or seven days with `--smoke` fail explicitly. Training has a configurable small synthetic run rather than a separate `--smoke` option.

## 2. PowerShell: selecting a saved model

From the repository root, while `.venv` is activated:

```powershell
# Selecting the most recent completed local model bundle.
$bundle = (
    Get-ChildItem "artifacts\synthetic_training" -Directory |
    Where-Object {
        Test-Path (Join-Path $_.FullName "bundle_manifest.json")
    } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
).FullName

Test-Path "$bundle\bundle_manifest.json"
finance-ml infer --help
```

`$bundle` is a PowerShell variable holding a **directory path**; `"$bundle"` substitutes that path in the command. Use the actual path printed after `finance-ml train`, not a placeholder. PowerShell variables live in the current shell session and must be set again in a new terminal. The folder must contain `best_model.keras`, `preprocessing_metadata.json`, and `bundle_manifest.json`.

### Legacy local-bundle provenance (one-time setup)

An example **local** model trained on **8 synthetic development accounts**, 60 days from `2026-01-01`, simulator seed `42`. The checkpoint stores the fitted feature transforms but **does not persist a verified population/seed record**. Declare these values explicitly before first account/date inference:

```powershell
finance-ml profile-init --bundle "$bundle" --development-accounts 8 --historical-days 60 --seed 42 --start-date 2026-01-01 --observed-through 2026-04-30 --max-new-accounts 1000
```

This writes `synthetic_provenance.json` **next to** the checkpoint. It is tied to the checkpoint's SHA-256 digest, does **not** change its manifest/checksums, and refuses to overwrite an existing profile. **Provenance is user-declared, not verified against the original training records.** Record different parameters only when supported by the actual training run; do not label an unrelated replay as the original training dataset. `--observed-through` is the defined synthetic experiment horizon, **not evidence that the local data were persisted to Databricks**.

The eight-account local model and the previously registered **2,000-account Databricks model v1 are different models with different development populations**. For this local bundle, account `ACC_00521` is *new*. In the registered-model notebook's separate population, `ACC_00521` is *known*. Do not switch those interpretations when reporting results.

## 3. Account/date inference syntax

```powershell
# One account; explicit start/end and selected model.
finance-ml infer --bundle "$bundle" --account-id ACC_00001 --start-date 2026-03-10 --end-date 2026-03-12 --dry-run

# Real smoke execution of exactly the specified two days and one account.
finance-ml infer --bundle "$bundle" --account-id 9 --start-date 2026-02-10 --end-date 2026-02-11 --smoke

# Full local inference on any explicitly selected accounts (not forced cohorts).
finance-ml infer --bundle "$bundle" --account-id 1 --account-id 2 --start-date 2026-03-10 --end-date 2026-03-20

# No observations after Apr 30 exist inside this replay profile by default:
finance-ml infer --bundle "$bundle" --account-id 521 --start-date 2026-05-01 --end-date 2026-05-03 --dry-run

# An opt-in synthetic scenario, NOT a feature-free future risk forecast:
finance-ml infer --bundle "$bundle" --account-id 521 --start-date 2026-05-01 --end-date 2026-05-03 --simulate-future
```

Input `--account-id 521` normalizes to `ACC_00521`; repeat the flag to select multiple accounts. Dates are **inclusive** and must use `YYYY-MM-DD`. Date spans may cross evaluation boundaries; each output row receives its own internally derived cohort.

| Dates / model-relative population | Existing in model development (1–8 for this bundle) | New to model development (9–1008 for this profile) |
|---|---|---|
| `2026-01-01`–`2026-03-01` | **Development period**; warning, excluded from independent holdout metrics | **Account Holdout** |
| `2026-03-02`–`2026-04-30` | **Date Holdout** | **Joint Holdout** |
| After `2026-04-30` | Opt-in **simulated extension** only | Opt-in **simulated extension** only |

Dates before January 1 are rejected. Dates after April 30 without `--simulate-future` are rejected. The opt-in extension is bounded to 60 days past the profile horizon. *Neither normal inference nor synthetic continuation writes Databricks tables.* The configured 1,000 new accounts in notebook 10 is a **cohort size**; the profile enforces that cap for reproducible local replay, not a fundamental simulator limitation.

### Causal endpoint policy

The inference CLI reads sequence length and ordered features from the **selected model**. The demonstrated local bundle has 7 timesteps and 31 features; the separately registered UC LSTM v1 has 7 timesteps and 41 features. For a requested date `t`, serving builds a trailing window ending at `t`; predictions use the model output at the **final sequence position**. No future-dated feature is allowed. The first six days of a new account cannot be scored without sufficient history; they appear in coverage output as `insufficient_context`. The original model-development validation instead **trimmed two positions at each end and aggregated overlapping windows**. Those macro-F1 metrics are not directly comparable to causal endpoint scores.

### Source and provenance boundaries

`--data-source synthetic --model-source local` reconstructs selected account histories **in memory** from the configured stateful simulator; it does not query Databricks. `--data-source databricks --model-source uc` reads **persisted Gold tables** and loads the **pinned UC model v1**, with separately documented cohort boundaries. A dry run checks local request constraints without connecting to Databricks or importing TensorFlow/MLflow. A local dry run cannot independently authenticate a user-declared simulator seed.

`--simulate-future` generates **future financial predictors** from preserved simulator latent-state and RNG snapshots, then classifies them. It does **not** perform future risk forecasting from April 30 alone. New-account replay follows the simulator's original fixed shared random-number stream through the profile's observed horizon; any extension then resumes the selected account's independent snapshot. This avoids changing historical observations when the requested end date changes. The LSTM's hidden/cell states are not persisted between CLI calls.

## 4. ASCII execution diagrams

**Training — dry run**

```text
finance-ml train --dry-run
    |
    +--> cli.py
    |      +--> config/config_loader.py          (YAML override)
    |      +--> config/training_config.py        (validate)
    |
    +--> print resolved configuration --> EXIT
    [No Spark / TensorFlow training / model registration]
```

**Training — local synthetic run**

```text
finance-ml train --data-source synthetic --config configs/training_synthetic.yaml ...
    |
    +--> cli.py / training_runner.py             (Spark ownership)
    |     +--> data/synthetic_training.py
    |           +--> data/generate_synthetic.py
    |           +--> lakehouse/build_silver.py --> lakehouse/build_gold.py
    |
    +--> pipelines/model_pipeline.py
          +--> model_dataset_pipeline.py        (fit training-only preprocessing)
          +--> models/experiment.py / training.py (LSTM + validation macro-F1)
          +--> models/local_bundle.py            (best_model + metadata + manifest)
                    |
                    +--> artifacts/.../run_<unique-id>/
    [No MLflow/UC logging for synthetic training]
```

**Inference — dry run / preflight**

```text
finance-ml infer --bundle "$bundle" --account-id 521 --start-date ... --end-date ... --dry-run
    |
    +--> cli.py
          +--> pipelines/account_inference_runner.py
                +--> models/local_bundle.py            (checksum/feature contract)
                +--> data/synthetic_provenance.py      (checkpoint-bound profile)
                +--> account/date/row-limit validation
                +--> display requested plan + warnings
    |
    +--> EXIT  [No Spark / no model load / no predictions]
```

**Inference — real smoke or full execution**

```text
finance-ml infer --bundle "$bundle" --account-id ... --start-date ... --end-date ...
    |
    +--> cli.py --> pipelines/account_inference_runner.py
          |
          +--> models/local_bundle.py             (verified Keras + fitted metadata)
          +--> data/synthetic_provenance.py       (population, seed, date horizon)
          +--> data/synthetic_replay.py           (stateful account replay)
          |       +--> data/generate_synthetic.py (original RNG/state equations)
          |       +--> [label-free observations]  [truth separately held]
          |
          +--> pipelines/training_runner.py      (Spark lifecycle ownership)
          +--> pipelines/inference_runner.py     (shared processing helpers)
          |       +--> lakehouse/build_silver.py --> build_gold.py
          |       +--> preprocessing/imputation.py / encoding.py / scaling.py
          |       +--> causal_endpoint_predictions() (trailing windows)
          |       +--> post-scoring label alignment / simulator metrics
          |
          +--> artifacts/inference/infer_<unique-id>/
                +--> predictions.csv
                +--> coverage.csv
                +--> inference_report.json
    [No training, UC writes, or remote model registration]
```

Smoke follows the **same real execution path** but validates that the user's selection contains at most two accounts and seven requested days; it does not change account IDs or dates.

## 5. Outputs and interpretation

Each output directory contains:

- `predictions.csv`: requested scorable account/date keys, class probabilities, selected class, `cohort`, `account_population`, and `observation_origin`.
- `coverage.csv`: every requested key, either `predicted` or `insufficient_context`, with the same row classification.
- `inference_report.json`: request, warnings, coverage, and metrics grouped by post-development cohort. **Development-period predictions are excluded from holdout evaluation**. Scores against simulator-generated labels are synthetic diagnostics, not production benchmarks.

```powershell
$run = 'artifacts\inference\infer_PASTE_REAL_DIRECTORY_HERE'
Get-ChildItem "$run"
Import-Csv "$run\predictions.csv" | Select-Object -First 5 | Format-Table
Get-Content "$run\inference_report.json"
```


### Inspecting evaluation reports in PowerShell

Inference results are saved in run-specific directories under
`artifacts/inference/`.

When evaluation is available, `inference_report.json` contains
an `evaluations` object with metrics grouped by cohort.

To inspect the most recent inference result:

```powershell
# Finding the most recent inference directory.
$run = (
    Get-ChildItem "artifacts\inference" -Directory -Filter "infer_*" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
).FullName

# Loading the JSON report.
$report = Get-Content "$run\inference_report.json" -Raw |
    ConvertFrom-Json

# Listing the evaluated cohorts.
$report.evaluations.PSObject.Properties.Name

# Displaying Account Holdout evaluation metrics.
$report.evaluations.account_holdout | Format-List

# Displaying individual class metrics.
$report.evaluations.account_holdout.per_class | Format-List *

# Displaying the confusion matrix.
$report.evaluations.account_holdout.confusion_matrix |
    ConvertTo-Json -Depth 5
```

Replace `account_holdout` with `date_holdout` or `joint_holdout`
to inspect a different cohort.

For local synthetic continuation experiments,
`simulated_extension` may also appear as an evaluation cohort.

The evaluation includes:

- Number of evaluated observations
- Accuracy
- Four-class macro-F1
- Per-class precision, recall, F1, and support
- Confusion matrix

The macro-F1 implementation assigns zero F1 to classes absent
from both actual and predicted labels. This convention is
documented in `docs/modeling_rules.md`.

For Databricks/Unity Catalog inference, ground-truth labels are
queried only when `--evaluate` is explicitly supplied.
Local synthetic replay can calculate simulator-based metrics
without that flag.

Metrics computed from a few observations serve as functional
checks, not reliable model-performance estimates.



## 6. Execution limits and future development

Local batch execution enforces at most 25 selected accounts, 20,000 requested/contextual rows, and 60 opt-in simulation-extension days; an account ID must lie within the declared synthetic population. Replay can still advance the random generator for unselected predecessor accounts to reproduce the original simulator stream. Original Spark session ownership rules are retained: **never stop caller-owned sessions, reuse existing sessions, stop newly created standalone sessions on success/failure, and preserve the original exception if cleanup also fails**.

Future milestones: store simulator provenance **automatically at synthetic training time** (rather than a user-declared legacy sidecar), consolidate synthetic data-quality injection and fixed feature-schema policy, and add a true forecasting/serving mode only with an explicit forecasting specification. The Databricks/UC v1 inference path has been verified through live read-only execution against the registered model and all three holdout cohorts. These smoke tests demonstrate functional integration, not comprehensive model-performance validation. Do not confuse these with the currently tested local synthetic classification workflow.



## 7. Model training: local and Databricks-backed workflows

The training CLI separates the source of training data from
the destination of model artifacts.

Both standalone training workflows execute Spark preprocessing
and TensorFlow/Keras training on the local computer.

| Data source | Execution | Model destination |
|---|---|---|
| Synthetic | Local Spark and Keras | Local Keras bundle |
| Databricks | Local Spark and Keras, using remote Gold data | Local bundle and optional MLflow logging |
| Databricks with UC registration enabled | Local Spark and Keras, using remote Gold data | MLflow LoggedModel and registered UC model |

The registered-model inference workflow can subsequently load
an existing UC model without retraining it.

### Local synthetic training

Dry-run validation:

```powershell
finance-ml train `
    --data-source synthetic `
    --config configs/training_synthetic.yaml `
    --synthetic-accounts 8 `
    --synthetic-days 60 `
    --synthetic-seed 42 `
    --dry-run
```

Execute training:

```powershell
finance-ml train `
    --data-source synthetic `
    --config configs/training_synthetic.yaml `
    --synthetic-accounts 8 `
    --synthetic-days 60 `
    --synthetic-seed 42
```

The pipeline generates synthetic account observations, processes
them through Silver and Gold, fits preprocessing on the training
partition, trains the LSTM, and saves the selected checkpoint
and its preprocessing metadata.

Synthetic CLI training explicitly disables MLflow logging and
Unity Catalog registration.

The current local synthetic workflow does not inject the
deliberate Bronze data-quality corruption used in the original
Databricks data-preparation workflow. Consequently, the fitted
input feature schema may differ between training runs.
This limitation is tracked separately from inference.

### Databricks-backed training

For standalone execution, select either the Databricks SQL
Connector or SQLAlchemy:

```powershell
finance-ml train `
    --data-source databricks `
    --query-method connector `
    --run-mode development `
    --dry-run
```

After reviewing the resolved configuration, remove `--dry-run`
to execute training.

The standalone CLI does not use local Spark to access Unity
Catalog tables directly. Instead, the connector retrieves
Databricks Gold data, which are processed through local Spark
and TensorFlow/Keras.

When training runs with `log_to_mlflow=true`, MLflow tracking
must be configured before execution. Development runs do not
register a model in Unity Catalog.

### Optional Unity Catalog model registration

UC registration is a separate, explicitly authorized
operation. It is not performed by the inference CLI.

For final training, an external YAML configuration may set:

```yaml
run_mode: final
query_method: connector
log_to_mlflow: true
register_in_uc: true
registered_model_name: finance_ml.models.lstm_risk_classifier
save_keras_checkpoint: true
```

For example, save this configuration as
`configs/training_uc_final.yaml`.

Validate it without execution:

```powershell
finance-ml train `
    --data-source databricks `
    --config configs/training_uc_final.yaml `
    --dry-run
```

When intentionally training and registering a new model version,
the actual execution additionally requires
`--allow-uc-registration`.

The environment must have appropriate Databricks credentials,
MLflow tracking configuration, and Unity Catalog permissions.

**Caution:** A final training run may create a new registered
model version. Do not run it merely to test inference against
an already registered model.

### Training execution architecture

```text
finance-ml train
    |
    +--> CLI configuration / validation
    |
    +--> Data source
    |      |
    |      +--> Synthetic generator
    |      |      +--> Local Spark Silver / Gold
    |      |
    |      +--> Databricks Gold table
    |             +--> SQL Connector / SQLAlchemy
    |             +--> Local Spark preprocessing
    |
    +--> Chronological train / validation / test split
    |
    +--> Fit training-only imputation, encoding and scaling
    |
    +--> Build seven-timestep sequences
    |
    +--> Train and evaluate LSTM
    |
    +--> Save run-specific local artifacts
    |
    +--> Optional MLflow logging
           |
           +--> Optional UC registration (explicitly authorized)
```

Saved preprocessing metadata are model-specific. Inference
must reproduce their exact feature order and transformations;
it must never refit preprocessing on requested observations.

## 8. Read-only Databricks / Unity Catalog inference

These commands retain the **same account/date interface**. There is no separate `--holdout` selector. The **selected model's population** determines whether an account is known or new.

```powershell
# Offline preflight: no query, MLflow load, Spark or model inference.
finance-ml infer --data-source databricks --model-source uc --account-id 521 --start-date 2026-03-10 --end-date 2026-03-12 --dry-run

# Actual remote-backed smoke run (two accounts and seven days maximum).
finance-ml infer --data-source databricks --model-source uc --account-id 521 --start-date 2026-03-10 --end-date 2026-03-12 --smoke

# One account, later dates; separately query simulated ground truth for metrics.
finance-ml infer --data-source databricks --model-source uc --account-id 2001 --start-date 2026-04-01 --end-date 2026-04-10 --evaluate

# Explicit SQLAlchemy option instead of the Databricks SQL Connector.
finance-ml infer --data-source databricks --model-source uc --query-method sqlalchemy --account-id 521 --start-date 2026-03-10 --end-date 2026-03-20
```

The **only currently supported registered cohort** is `finance_ml.models.lstm_risk_classifier` version `1`. The CLI refuses other versions rather than applying v1's account/date definitions to an unrelated model. The registered-model loader checks the version's preprocessing run, metadata SHA-256, feature order and model input shape; UC v1 is expected to have **7 timesteps, 41 features, 4 classes**. In contrast, the local example checkpoint has **31 features** and an eight-account development population.

The persisted UC model population and dates come from `notebooks/10_build_future.ipynb` and `docs/holdout_data_pipeline.md`:

| UC account numbers | Jan 1–Mar 1, 2026 | Mar 2–Apr 30, 2026 |
|---|---|---|
| 1–2000 | Development period (not independent evaluation) | **Date Holdout** |
| 2001–3000 | **Account Holdout** | **Joint Holdout** |

Dates before January 1 and after April 30 are rejected. The Databricks branch **never accepts `--simulate-future`, `--profile`, or `--bundle`**; synthetic continuation is not a fallback for unavailable persisted data. The maximum 25 selected accounts/20,000 requested observations and smoke limit (2 accounts/7 days) remain in effect. No original tables or registered model are changed.

### UC execution modules: dry-run

```text
finance-ml infer --data-source databricks --model-source uc ... --dry-run
    |
    +--> cli.py
         +--> data/databricks_holdout.py
              +--> Validate account IDs, date range, fixed v1 population
              +--> Select Gold table names and predicted cohorts
    |
    +--> Print plan and warnings --> EXIT
    [No remote connection, Spark, model load or local result files]
```

### UC execution modules: real smoke or full run

```text
finance-ml infer --data-source databricks --model-source uc ...
    |
    +--> cli.py --> pipelines/databricks_inference_runner.py
          |
          +--> models/loading.py
          |      +--> MLflow / pinned Unity Catalog model v1
          |      +--> Verify registered preprocessing artifact SHA-256
          |      +--> Get 7 x 41 model input contract
          |
          +--> data/databricks_holdout.py
          |      +--> Build bounded SELECTs for required source FEATURES only
          |      +--> data/query.py (connector OR sqlalchemy)
          |      +--> Persisted Gold (development / date / new account)
          |      +--> Include preceding context, reject missing requested rows
          |
          +--> Typed local Spark Gold DataFrame
          |      +--> pipelines/inference_runner.py: apply_saved_preprocessing
          |      +--> preprocessing/imputation.py, encoding.py, scaling.py
          |      +--> causal_endpoint_predictions (no future inputs)
          |
          +--> Optional --evaluate (after prediction only)
          |      +--> Read separate UC truth tables (Date/Account/Joint)
          |      +--> Exclude development-period metric computation
          |
          +--> artifacts/inference/infer_<unique-id>/
                 +--> predictions.csv / coverage.csv / inference_report.json
    [No UC table writes, model registration, training or silent simulation]
```

The Date Holdout Gold may require **up to six historical preceding dates** from the development Gold table. The account-holdout Gold already contains both historical and future rows for each new account; only relevant context/selection is read. For the **first six days of a genuinely new account**, coverage may report `insufficient_context` instead of fabricating a seven-step window. The policy is `causal_endpoint_v1`, different from original overlap-trimmed training metrics.

**Authentication:** Keep the existing `.env` file private, with the Databricks SQL warehouse hostname, HTTP path, and access token; the existing configuration loader supplies query credentials. For registered-model loading, MLflow uses the Databricks host/token and `databricks` tracking plus `databricks-uc` registry URIs. A standalone local Spark session is used for saved feature transformations, and is stopped only if created by the runner. The SQL warehouse and Unity Catalog model may require distinct workspace permissions; neither can be verified by an offline dry run.

**Date-key normalization:** Persisted Gold/truth timestamps are interpreted in UTC and normalized to UTC midnight. Local Spark represents observation keys as calendar dates; non-midnight source timestamps are rejected. This prevents timezone-aware SQL Connector timestamps from breaking pandas joins and avoids workstation daylight-saving shifts.

**Evaluation:** `--evaluate` is opt-in: it queries `finance_ml.gold.account_labels_temporal_holdout` or `finance_ml.gold.account_labels_account_holdout` *after* model scoring, joins by `(account_id, date)`, and reports synthetic-diagnostic metrics only. Without it, the CLI does not read labels for evaluation. The original historical Gold table may contain labeled columns, but the real inference SELECT requests **only the ordered numeric and categorical input source columns** (plus keys).

**Security and limitations:** SQL queries are allowlisted to three Gold tables and two truth tables, with strict account/date validation. No DDL, `INSERT`, or `OVERWRITE` operations are executed. Remote inference is bounded and driver-oriented; it is not a live serving endpoint. First test read access, MLflow loading and Spark execution with a single-account smoke run. Preserve the 31/41-feature discrepancy as a distinct training-schema improvement rather than changing registered-model preprocessing.
