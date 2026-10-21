# FastAPI + Streamlit: developer and user guide

## 1. Purpose and scope

The platform separates **interactive presentation** (Streamlit) from **validated inference** (FastAPI). A user uploads *raw, unlabeled* account-day records, the backend processes the uploaded data using the existing Silver/Gold data pipeline and the saved preprocessing contract, and the local Keras LSTM returns a risk classification for eligible account dates. Streamlit plots **actual API results**, not example or placeholder predictions.

This guide covers the current **local CSV upload + local Keras bundle** route. It does not describe production authentication, remote hosting, automatic data ingest, UC serving, or future-risk forecasting. See the [repository README](../README.md) for a fresh-clone quick start and [CLI guide](cli_guide.md) for separate training/batch-inference workflows.

## 2. Component ownership

```text
User / web browser (http://127.0.0.1:8501)
    |
    | Upload raw, unlabeled .csv; click "Generate predictions"
    v
src/finance_ml/dashboard/app.py          [Streamlit + Plotly + httpx]
    |                 |
    | GET             | POST multipart/form-data (CSV bytes)
    | /ready          | /uploads/validate, /predict/upload
    | /model          v
    +--------> http://127.0.0.1:8000 [Uvicorn + FastAPI]
                         |
                         v
                   api/main.py
                  /         \
          /ready, /model    upload endpoints
                |                |
          api/model_service.py   api/csv_validation.py
                |                |
         LocalModelBundle        v
         verified Keras    api/spark_runtime.py  [lazy local Spark]
           + metadata           |
                    +-----------+
                                v
                     api/prediction_pipeline.py
                                |
                                v
                       api/upload_pipeline.py
                        /                \
                Silver cleaning       Gold feature engineering
                        \                /
                         +--------------+
                                |
                     saved preprocessing (NO refit)
                                |
                     causal endpoint inference
                                |
                  predictions, coverage, context
                                |
                     api/result_store.py
                                |
               +----------------+----------------+
               |                                 |
     run-specific CSV/JSON                latest_*.csv/json
               |
               +--- response JSON ---> Streamlit session_state
                                             |
                               risk charts, tables, CSV download
```

**Boundary rule:** Keep training, feature generation, fitted transformations, saved model loading, and data validation **out of** Streamlit. The backend owns all inference contracts. The frontend makes HTTP requests and renders results.

## 3. Files and responsibilities

| Path | Responsibilities |
|---|---|
| `src/finance_ml/api/main.py` | FastAPI application, startup/shutdown lifespan, endpoint handlers, upload-size checks, HTTP error handling |
| `src/finance_ml/api/schemas.py` | Pydantic request and response models, typed predictions, summary and coverage data |
| `src/finance_ml/api/csv_validation.py` | Exact raw CSV columns, ISO dates, numeric parsing, target exclusion, row limits |
| `src/finance_ml/api/model_service.py` | Load and retain integrity-verified `.keras` model bundle with its persisted preprocessing metadata |
| `src/finance_ml/api/spark_runtime.py` | Lazily create/reuse/close the local Spark session; serialize Spark inference calls |
| `src/finance_ml/api/upload_pipeline.py` | Convert validated pandas raw observations to Spark; reuse Silver and Gold transformations; compute per-account context |
| `src/finance_ml/api/coverage.py` | Assess account sequence eligibility and missing calendar days |
| `src/finance_ml/api/prediction_pipeline.py` | Apply saved model preprocessing; use causal endpoint predictions; assemble warnings, coverage, and response |
| `src/finance_ml/api/result_store.py` | Validate result contracts; persist run-specific and stable latest CSV/JSON artifacts |
| `src/finance_ml/dashboard/app.py` | Browser UI; readiness/model checks; CSV upload; validation and inference requests; stateful results; Plotly charts and CSV download |
| `scripts/06_create_sample_upload_csv.py` | Generate a reproducible, unlabeled, raw-data CSV to exercise the upload contract |
| `.env.example` | Template including `FINANCE_ML_API_BUNDLE_DIR` (no model files or secrets) |
| `tests/test_api_*.py` | Backend endpoint and pipeline tests, including mocked services |
| `tests/test_dashboard_app.py` | UI smoke tests, HTTP-client tests and rendering-contract tests without Spark or a real API process |

Supporting ML components under `src/finance_ml/models/` and `src/finance_ml/pipelines/` implement saved model loading and causal endpoint prediction; `src/finance_ml/lakehouse/` retains the Silver/Gold transformations.

## 4. Service startup and configuration

The saved model bundle is a **directory**, not a single `.keras` file; it contains a Keras checkpoint, the exact feature order, imputation/encoding/scaling metadata, and a checksum/signature manifest. New clones **must train locally or acquire a trusted bundle separately**; generated model artifacts are excluded from Git.

At backend startup, `api/main.py` loads `.env` from the project root and reads:

```dotenv
FINANCE_ML_API_BUNDLE_DIR=D:/Projects/finance-timeseries-ml-platform/artifacts/synthetic_training/<actual-run-directory>
```

`ModelService.from_directory()` runs once in the lifespan when configured. `/health` can succeed without a model, whereas `/ready`, `/model`, and `/predict/upload` require a successfully loaded model. `SparkRuntime` is created at startup, but the actual Spark session is **lazy**, initialized on first prediction. During shutdown the backend closes a Spark session it owns.

Streamlit uses `FINANCE_ML_API_URL` from its **process environment** if set (default `http://127.0.0.1:8000`). The current dashboard does not load this setting from `.env` itself. For a custom URL in PowerShell:

```powershell
$env:FINANCE_ML_API_URL = "http://127.0.0.1:8000"
```

**Run from the repository root**, in two terminals:

```powershell
# Terminal A: backend
python -m uvicorn finance_ml.api.main:app --host 127.0.0.1 --port 8000

# Terminal B: frontend
python -m streamlit run src/finance_ml/dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

The project environment must contain the `api`, `dashboard`, and `dev` extras for development; initialize once using `uv sync --locked --extra api --extra dashboard --extra dev`.

## 5. API contract

| Method | Endpoint | Purpose | Successful outcome |
|---|---|---|---|
| `GET` | `/health` | Process liveness independent of loaded model | `200`, `{"status":"ok"}` |
| `GET` | `/ready` | Model-serving readiness | `200`, `{"status":"ready"}` |
| `GET` | `/model` | Model metadata without exposing raw bundle path | `200`, JSON with model source, sequence length, feature/class counts, SHA-256 |
| `POST` | `/uploads/validate` | CSV syntax/schema inspection **without inference** | `200`, counts for rows, accounts, duplicates |
| `POST` | `/predict/upload` | Validate CSV, apply Silver/Gold and saved preprocessing, infer and persist outputs | `200`, JSON predictions, coverage, summary, warnings, run ID/time |

The POST routes receive `multipart/form-data` with a file field called **`file`**. The dashboard sends it using HTTPX's `files={"file": (filename, content, "text/csv")}`.

Upload contract (exactly these 12 columns; order arbitrary):

```text
account_id,date,account_type,balance,cash_inflow,cash_outflow,transaction_count,credit_utilization,amount_due,amount_paid,payment_ratio,days_past_due
```

- Date format: `YYYY-MM-DD`; valid ISO calendar days.
- `account_id` cannot be empty; `account_type` can be missing for Silver cleanup.
- Numeric values must parse as numbers or missing and must not include infinities.
- `risk_state`/`risk_state_name` **must not be supplied**; labels must not leak into inference.
- Maximum input file: 10 MiB; maximum rows: 20,000.
- Duplicate `(account_id, date)` pairs are reported at validation, then processed by the existing cleaning pipeline; they are not assumed to be independent timepoints.

### HTTP status meanings

- **200:** request succeeded.
- **413:** uploaded file exceeds 10 MiB.
- **415:** file extension is not `.csv`.
- **422:** invalid or empty CSV, invalid schema/types/date, or invalid request.
- **503:** model unavailable (`/ready`, `/model` and prediction route).
- **500:** unexpected processing/persistence failure; backend logs contain details, while the HTTP response remains generic.

FastAPI's generated interactive API page is available at **`http://127.0.0.1:8000/docs`**.

## 6. End-to-end upload and inference sequence

```text
Browser                Streamlit (8501)       FastAPI (8000)       Spark / Model
   |                          |                     |                    |
   | Open dashboard           |                     |                    |
   +------------------------->| GET /ready          |                    |
   |                          +-------------------->| check model        |
   |                          |<--------------------+                    |
   |                          | GET /model          |                    |
   |                          +-------------------->| describe bundle    |
   |                          |<--------------------+                    |
   | Upload CSV and click     |                     |                    |
   +------------------------->| POST /uploads/validate (multipart)         |
   |                          +-------------------->| check raw schema   |
   |                          |<--------------------+ counts             |
   |                          | POST /predict/upload|                    |
   |                          +-------------------->| validate again     |
   |                          |                     +------------------->| Silver/Gold
   |                          |                     |                    | saved preprocess
   |                          |                     |                    | trailing windows
   |                          |                     |<-------------------+ probabilities
   |                          |                     | save CSV/JSON      |
   |                          |<--------------------+ JSON + run_id      |
   |<-------------------------+ charts and table    |                    |
   | Download predictions CSV |                     |                    |
```

**Important:** `/uploads/validate` and `/predict/upload` each validate the uploaded bytes. Validation is not a security or authorization boundary for a later upload; inference must validate again.

## 7. Prediction policy and output semantics

The current model is a **unidirectional seq2seq LSTM**, typically with `sequence_length = 7` and `n_features = 41`. For the requested account/date, it builds a trailing window ending on that date and selects **the final model-output position only**. It makes a **same-date classification**, not a forecast of an unobserved future date. The first `L-1` account observations cannot be scored without enough context; the API reports them as `insufficient_context` rather than inventing results. Calendar gaps are counted separately. Full details: [sequence prediction policies](sequence_prediction_policies.md).

The API JSON response includes:

```text
run_id, created_at_utc
summary                 counts of accounts, predictions, insufficient context
predictions[]           account_id, date, predicted_class,
                        predicted_state_name, prob_<each of four classes>
coverage[]              account_id, date, predicted / insufficient_context
account_context[]       account-level historical context, gaps, eligibility
warnings[]              incomplete context / calendar gaps
data_source             uploaded_csv
cohort_status           unverified_external_upload
sequence_length         sourced from bundle metadata
```

The `prob_*` fields are four class probabilities from the model's softmax output and sum to approximately one per prediction. They should **not** be described as calibrated probabilities without separate calibration evidence. An uploaded account ID is not proof of membership in the training cohort.

The dashboard provides counts, the most recent account classification, a time-indexed risk-state plot, four class-probability curves, a risk-class distribution chart, coverage table, and a downloadable predictions CSV. Account selection reruns Streamlit, but the result is kept in `st.session_state`; it does not trigger inference again solely because the selected account changed.

## 8. Result persistence and BI contract

`api/result_store.py` writes to `artifacts/api_inference/`:

```text
artifacts/api_inference/
|-- latest_predictions.csv
|-- latest_coverage.csv
|-- latest_account_context.csv
|-- latest_summary.json
`-- runs/
    `-- run_<timestamp>_<id>/
        |-- predictions.csv
        |-- coverage.csv
        |-- account_context.csv
        `-- summary.json
```

Each CSV contains `run_id` in addition to the data fields. `summary.json` records model SHA-256, data source, cohort status, sequence length, and run-level counts/warnings. Run directories preserve historical executions; stable latest files make downstream Power BI ingestion convenient. These files are **ignored by Git**. The current Streamlit interface consumes the current HTTP JSON response, not the latest files, so different users are not coupled to one shared latest result.

## 9. Testing strategy

- **API unit/contract:** FastAPI `TestClient`, monkeypatched `FINANCE_ML_API_BUNDLE_DIR`, mocked `ModelService` and `SparkRuntime`, schema/status/error behavior. Tests that require no model must set the env var to an explicit empty string to prevent `.env` fallback.
- **Dashboard unit:** Patch HTTPX to test requests, error handling, and `multipart/form-data` without opening network ports. Patch Streamlit functions to verify rendering counts/plots/download input from a known response.
- **Dashboard smoke:** `st.testing.v1.AppTest` launches the actual Streamlit script with a simulated backend-ready/offline response; it requires no browser or live FastAPI service.
- **Manual end to end:** Run local FastAPI and Streamlit, generate sample raw CSV, upload, check run ID, number of predictions, data download, and persisted output paths. A mocked dashboard test alone **does not** prove real Spark/Keras inference.

```powershell
python -m pytest -q tests/test_dashboard_app.py
python -m pytest -q
```

GitHub Actions must install `.[api,dashboard,dev]` to import the Streamlit module when running all tests.

## 10. Deployment and security limitations

This is a **local learning/portfolio application**, not a secure hosted banking service. The current API does **not** implement user authentication, rate limiting, TLS termination, authorization, data-retention policy, or secure multi-tenant storage. Bind both servers to `127.0.0.1` and do not expose real customer or sensitive financial records. Production deployment would require authentication, HTTPS, secrets management, stricter input limits, secure persistence, access controls, observability, and workload/resource isolation.
