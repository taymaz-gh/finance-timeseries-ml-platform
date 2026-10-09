# Model Selection and Unity Catalog Registration

```text

      The ending section of notebook 08:
        Best Model Config (Winner)
                    │
                    ▼
           FINAL MODEL trained,
        persisted by two mechanism
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
 best_model.keras       MLflow LoggedModel
   Workspace file          m-a360...
          │                    │
          │                    │
          ▼                    ▼
(load with Keras  ◄ vs ►   load with MLflow)  ---►  ((( Verify against eachother )))
                            _______
                               │
                               ▼
                     [NEXT in Notebook 09: register to UC]
                               │
                               ▼
                     UC Registered Model
                     catalog.schema.model
```





## 1. Architecture

Model training, model selection, and model registration are separate operations.

```text
                    NOTEBOOK 08
              Model training and logging
                           |
                +----------+----------+
                |                     |
                v                     v
         Keras checkpoint       MLflow LoggedModel
         best_model.keras       models:/m-...
                                      |
                                      v
                              MODEL SELECTION
                              registry.py
                                      |
                          +-----------+-----------+
                          |                       |
                          v                       v
                    Final-run tag           Explicit model URI
                    final_model=true        models:/m-...
                          |                       |
                          +-----------+-----------+
                                      |
                                      v
                           Selection validation
                           - READY model
                           - Expected model name
                           - Consistency if both
                                      |
                                      v
                    NOTEBOOK 09: MODEL REGISTRATION
                                      |
                                      v
                          Unity Catalog Registry
                  finance_ml.models.lstm_risk_classifier
                                      |
                                      v
                           Registered version
                               Version N
                                      |
                                      v
                          Inference / Deployment
```

## 2. Separation of responsibilities

| Component | Responsibility |
|---|---|
| `notebooks/08_model_selection_mlflow.ipynb` | Train, evaluate, persist, and log the final model |
| `src/finance_ml/models/registry.py` | Resolve the selected LoggedModel URI and, separately, provide model-registration operations |
| `tests/test_registry.py` | Test selection, validation, and registration behavior |
| `notebooks/09_register_model_uc.ipynb` | Execute the reproducible registration workflow |
| Unity Catalog | Store registered model names, versions, and metadata |

**Model selection** identifies which persisted model should be registered.

**Model registration** creates a versioned entry for that model in Unity Catalog.

Registration must not require retraining.

## 3. Model selection logic

The function `resolve_logged_model_uri()` supports two independent methods and their combination.

```text
                          START
                            |
                            v
                 resolve_logged_model_uri()
                            |
               +------------+------------+
               |                         |
               v                         v
        model_uri provided?       use_final_tag=True?
               |                         |
               v                         v
       Validate LoggedModel       Find latest final-tagged run
       - URI format              - final_model=true
       - Model exists            - Find associated LoggedModel
       - Status READY            - Expected model name
               |                 - Status READY
               |                         |
               +------------+------------+
                            |
                            v
                  Which methods were used?
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
          URI only       Tag only       Both methods
              |             |             |
              v             v             v
          Return URI     Return URI     Compare model IDs
                                          |
                                    +-----+-----+
                                    |           |
                                    v           v
                                  Equal      Different
                                    |           |
                                    v           v
                                Return URI    ValueError

       Neither method provided -> ValueError
       Missing/ambiguous final model -> LookupError
```

### Selection rules

| `model_uri` | `use_final_tag` | Result |
|---|---|---|
| Provided | `False` | Validate and return explicit URI |
| Not provided | `True` | Discover the final model automatically |
| Provided | `True` | Require both methods to identify the same model |
| Not provided | `False` | Raise `ValueError` |

When tag-based selection is enabled, `experiment_id` is required.

If multiple runs have `final_model=true`, the current implementation selects the most recent run, then requires exactly one matching READY LoggedModel with the expected name.

For automated production workflows, passing an exact approved run ID is preferable to silently selecting the most recent run.

## 4. Usage examples

### Method A — Automatic selection using the final-model tag

```python
from finance_ml.models.registry import resolve_logged_model_uri

uri_a = resolve_logged_model_uri(
    experiment_id="2355252254662315",
    use_final_tag=True,
)
```

```text
Experiment
    |
    v
Runs with final_model=true
    |
    v
Most recent matching run
    |
    v
READY LoggedModel named lstm_model
    |
    v
Return models:/m-...
```

### Method B — Explicit model URI

```python
uri_b = resolve_logged_model_uri(
    model_uri="models:/m-a36002eda96e4677b3f6aa36818ed715",
)
```

```text
Explicit URI
    |
    v
Validate URI and model status
    |
    v
Return URI
```

This method does not require tag-based discovery.

### Method C — Combined selection with consistency checking

```python
uri_c = resolve_logged_model_uri(
    experiment_id="2355252254662315",
    model_uri="models:/m-a36002eda96e4677b3f6aa36818ed715",
    use_final_tag=True,
)
```

```text
Explicit URI ----------+
                       |
                       v
                   Compare IDs ---- Different ---> ValueError
                       ^
                       |
Final-tagged model ----+
                       |
                     Equal
                       |
                       v
                 Return model URI
```

This method is useful when the caller expects a specific model and also wants to verify that it matches the current final-model selection.

## 5. Unity Catalog registration

The `register_logged_model()` function registers an already-selected MLflow LoggedModel in Unity Catalog.

It is independent of `resolve_logged_model_uri()`: model selection determines *which model* to register, while registration determines *whether a new registered version is needed*.

### Registration workflow

```text
resolve_logged_model_uri()
          |
          v
   models:/m-...
          |
          v
register_logged_model()
          |
          +-- Validate UC model name
          |   catalog.schema.model
          |
          +-- Validate source LoggedModel
          |   - Exists
          |   - Status READY
          |
          v
   Search registered versions
          |
          v
   Already registered?
          |
     +----+----+
     |         |
    YES        NO
     |         |
     v         v
 Return       Register source
 existing     LoggedModel in UC
 version          |
                  v
             Create version N
                  |
                  v
             Set provenance tag
             source_logged_model_id
                  |
                  v
             Return version N
```

### Function interface

```python
from finance_ml.models.registry import register_logged_model

registered_version = register_logged_model(
    model_uri=selected_model_uri,
    registered_model_name="finance_ml.models.lstm_risk_classifier",
)
```

**Inputs**

| Parameter | Purpose |
|---|---|
| `model_uri` | URI of the selected MLflow LoggedModel |
| `registered_model_name` | Fully qualified Unity Catalog model name |
| `client` | Optional MLflow client, injectable for testing |

**Output:** An MLflow ModelVersion object representing either an existing matching version or a newly registered version.

### Duplicate-registration prevention

```text
Request: models:/m-abc123
             |
             v
     UC Registered Model
             |
       +-----+-----+
       |           |
       v           v
    Version 1   Version 2
    m-other     m-abc123
       |           |
       +-----+-----+
             |
             v
       Match found
             |
             v
    Return Version 2
    (no new version)
```

The current implementation checks existing versions for either:

- A matching `source_logged_model_id` version tag.
- A matching model source URI.

For newly registered versions, it records the source LoggedModel ID using the `source_logged_model_id` tag.

This provides duplicate-registration prevention for sequential executions. Concurrent registration attempts may still create duplicate versions unless additional coordination is introduced.

### Registry configuration

Registration is intended for the Databricks Unity Catalog registry:

```python
import mlflow

mlflow.set_registry_uri("databricks-uc")
```

The registry client and registration call must use the same registry destination.

### Example: selection followed by registration

```python
from finance_ml.models.registry import (
    register_logged_model,
    resolve_logged_model_uri,
)

selected_model_uri = resolve_logged_model_uri(
    experiment_id="2355252254662315",
    use_final_tag=True,
)

registered_version = register_logged_model(
    model_uri=selected_model_uri,
    registered_model_name="finance_ml.models.lstm_risk_classifier",
)

print("Registered model:", registered_version.name)
print("Version:", registered_version.version)
```

Registration does not retrain the model. Repeated execution should reuse an existing matching version when one can be identified.

**Implementation status:** The model-selection function has been verified against the live MLflow experiment. The registration function is implemented, but its tests and registry-configuration checks are still being completed. Live Unity Catalog registration has not yet been performed.

## 6. Reproducibility

```text
Version-controlled source code + SQL
                 |
                 v
          Git feature branch
                 |
                 v
       Automated unit tests
                 |
                 v
         Databricks Notebook 09
                 |
                 v
       Unity Catalog registration
                 |
                 v
       Verified registered model
```

The model's URI is discovered or provided as an input; it is not hardcoded inside the reusable registry implementation.

The notebook and registration code can be rerun without performing model training, and registration should avoid duplicate model versions.

**Note:** The examples use identifiers from the current development experiment. Other deployments should supply their own experiment identifiers and model URIs.