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
$bundle = 'artifacts\synthetic_training\run_20261014T061609072027Z_ca1a03e520'
Test-Path "$bundle\bundle_manifest.json"
finance-ml infer --help
```

`$bundle` is a PowerShell variable holding a **directory path**; `"$bundle"` substitutes that path in the command. Use the actual path printed after `finance-ml train`, not a placeholder. PowerShell variables live in the current shell session and must be set again in a new terminal. The folder must contain `best_model.keras`, `preprocessing_metadata.json`, and `bundle_manifest.json`.

### Legacy local-bundle provenance (one-time setup)

Our existing light **local** model trained on **8 synthetic development accounts**, 60 days from `2026-01-01`, simulator seed `42`. The checkpoint stores the fitted feature transforms but **does not persist a verified population/seed record**. Declare these values explicitly before first account/date inference:

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

The local model accepts seven consecutive timesteps and 31 features (for the demonstrated bundle). For a requested date `t`, serving builds a trailing window ending at `t`; predictions use the model output at the **final sequence position**. No future-dated feature is allowed. The first six days of a new account cannot be scored without sufficient history; they appear in coverage output as `insufficient_context`. The original model-development validation instead **trimmed two positions at each end and aggregated overlapping windows**. Those macro-F1 metrics are not directly comparable to causal endpoint scores.

### Source and provenance boundaries

`--data-source synthetic` is currently supported. It reconstructs selected account histories **in memory** from the configured stateful simulator, with deterministic known-account history and new-account replay; it does not query the persisted Databricks holdout tables. Databricks-backed inference and loading the registered UC model require a separate implementation and are explicitly rejected by this CLI. A local dry run can verify the bundle, profile, account IDs, requested range, and guards but cannot independently authenticate a user-declared simulator seed.

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

## 6. Execution limits and future development

Local batch execution enforces at most 25 selected accounts, 20,000 requested/contextual rows, and 60 opt-in simulation-extension days; an account ID must lie within the declared synthetic population. Replay can still advance the random generator for unselected predecessor accounts to reproduce the original simulator stream. Original Spark session ownership rules are retained: **never stop caller-owned sessions, reuse existing sessions, stop newly created standalone sessions on success/failure, and preserve the original exception if cleanup also fails**.

Future milestones: store simulator provenance **automatically at synthetic training time** (rather than a user-declared legacy sidecar), implement read-only Databricks query paths and registered UC-model loading with **model-specific** cohort provenance, and add a true forecasting/serving mode only with an explicit forecasting specification. Do not confuse these with the currently tested local synthetic classification workflow.
