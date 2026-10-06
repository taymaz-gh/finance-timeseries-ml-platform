# Model Selection and Controlled Experimentation

## 1. Purpose

This document defines the procedure for comparing and selecting sequence-classification models in the project.

The current model-development stage starts from the established LSTM baseline and investigates a small set of controlled architectural and optimization configurations.

The objective is to identify a well-performing model while maintaining a consistent data-processing and evaluation procedure and avoiding the use of the test set for model-selection decisions.

The selected model will subsequently be used for final test evaluation and model persistence.

---

## 2. Baseline model

The current reference model is a single-layer LSTM sequence classifier with:

- LSTM units: `(64,)`
- dropout rate: `0.30`
- Adam learning rate: `1e-3`
- sequence-to-sequence output with one class-probability vector per timestep;
- sparse categorical cross-entropy loss;
- timestep-level accuracy;
- aggregated timestep-level macro-F1 for model selection.

The baseline provides the reference against which alternative configurations are compared.

---

## 3. Controlled comparison principle

Candidate models must be compared under the same experimental conditions.

Unless a configuration explicitly investigates one of these factors, the following must remain unchanged:

- model-dataset construction;
- training, validation, and test partitions;
- feature preprocessing;
- training-only preprocessing statistics;
- sequence length;
- sequence stride;
- target encoding;
- boundary-trimming rule;
- overlapping prediction aggregation;
- true-label alignment;
- evaluation population;
- metric definitions.

The same validation and test evaluation procedures described in `docs/modeling_rules.md` are used for every candidate.

Consequently, differences in validation performance should primarily reflect differences in the model configuration or optimization settings rather than differences in data preparation or evaluation.

---

## 4. Primary model-selection metric

The primary model-selection metric is **validation Macro-F1**.

Specifically, the selection metric is the aggregated timestep-level `val_macro_f1` produced during training.

This metric is calculated after:

1. generating per-timestep softmax probabilities;
2. trimming sequence-boundary predictions;
3. aggregating overlapping predictions at the original `(group, time)` level;
4. determining one predicted class for each retained original timestep;
5. calculating Macro-F1 across the target classes.

Macro-F1 is preferred over accuracy for model selection because it assigns equal weight to the target classes rather than allowing frequent classes to dominate the summary metric.

A higher validation Macro-F1 indicates better performance.

---

## 5. Early stopping

Training uses validation Macro-F1 for early stopping.

The best model weights are restored according to the highest observed validation Macro-F1.

The test set is not involved in determining:

- the stopping epoch;
- the selected epoch;
- the selected architecture;
- the selected hyperparameters.

The epoch corresponding to the highest validation Macro-F1 is recorded as part of the experiment results.

---

## 6. Overfitting assessment

Model selection is not based solely on the highest numerical validation Macro-F1.

Training dynamics are also inspected using the learning curves for:

- loss;
- accuracy;
- Macro-F1.

In particular, the following are examined:

### Training versus validation performance

A growing separation between training and validation performance can indicate overfitting.

For Macro-F1, particular attention is paid to situations where:

- training Macro-F1 continues to increase;
- validation Macro-F1 stops improving or decreases.

### Loss behaviour

A pattern in which training loss continues to decrease while validation loss increases can provide additional evidence of overfitting.

### Stability of validation performance

Small fluctuations in validation Macro-F1 are not by themselves considered evidence of severe overfitting.

The selected model should provide strong validation performance without unnecessary model complexity or clearly deteriorating validation performance.

---

## 7. Candidate configurations

The initial model-selection experiment consists of a small number of deliberately chosen configurations.

| Configuration | LSTM units | Dropout | Learning rate | Purpose |
|---|---|---:|---:|---|
| A | `(64,)` | 0.30 | `1e-3` | Baseline |
| B | `(64,)` | 0.50 | `1e-3` | Investigate stronger regularization |
| C | `(128,)` | 0.30 | `1e-3` | Investigate increased model capacity |
| D | `(64, 32)` | 0.30 | `1e-3` | Investigate increased depth |
| E | `(64,)` | 0.30 | `5e-4` | Investigate a lower learning rate |

The configurations are intended to test distinct hypotheses rather than perform an unrestricted hyperparameter search.

All candidate configurations use the same random seed (`42`) for
reproducible model initialization and stochastic training.


### Configuration A -- Baseline

Configuration A reproduces the established LSTM baseline and provides the reference validation performance.

### Configuration B -- Stronger regularization

The dropout rate is increased from `0.30` to `0.50` while keeping the architecture and learning rate unchanged.

This configuration investigates whether stronger regularization improves validation performance or reduces the observed training–validation gap.

### Configuration C -- Increased capacity

The number of units is increased from `64` to `128`.

This configuration investigates whether the baseline is capacity-limited.

### Configuration D -- Increased depth

A second LSTM layer is introduced using `(64, 32)` units.

This configuration investigates whether hierarchical temporal representations improve performance.

### Configuration E -- Lower learning rate

The learning rate is reduced from `1e-3` to `5e-4`.

This configuration investigates whether a smaller optimization step improves convergence or validation performance.

---

## 8. MLflow experiment tracking

Each candidate configuration is recorded as a separate MLflow run.

MLflow is used to provide a persistent record of the model-selection experiments rather than relying only on notebook output.

### Parameters

Each run should record the configuration parameters, including:

- LSTM layer configuration;
- dropout rate;
- learning rate;
- batch size;
- maximum number of epochs;
- early-stopping patience;
- sequence length;
- boundary width;
- number of input features;
- number of target classes.

### Metrics

Training and validation metrics should be recorded for each experiment.

The principal selection metrics include:

- best validation Macro-F1;
- epoch corresponding to the best validation Macro-F1;
- training Macro-F1 at the selected epoch;
- validation accuracy at the selected epoch;
- validation loss at the selected epoch.

The final test metrics are recorded only for the selected final model.

### Artifacts

Each experiment should retain the training artifacts generated by the experiment framework, including:

- `training_history.csv`;
- `learning_curve_loss.png`;
- `learning_curve_accuracy.png`;
- `learning_curve_macro_f1.png`.

The experiment configuration and associated artifacts should therefore be traceable to the corresponding MLflow run.

---

## 9. Test-set policy

The test set is reserved for final evaluation.

It must not be used to:

- select the architecture;
- select hyperparameters;
- determine dropout;
- determine learning rate;
- determine the number of layers;
- determine the number of units;
- determine the stopping epoch;
- choose between candidate models.

Test Macro-F1 and test accuracy should therefore not be used to rank the candidate configurations.

During model selection, candidate configurations are compared using the training and validation partitions only.

After the final configuration has been selected, the selected model is evaluated on the test partition once to obtain the final reported test performance.

---

## 10. Model-selection procedure

The complete procedure is:

```text
Prepared model dataset
        |
        v
Fixed train / validation / test partitions
        |
        v
Candidate configuration
        |
        v
Build and compile LSTM
        |
        v
Train on training partition
        |
        +----> Training metrics
        |
        +----> Validation metrics
        |
        v
Early stopping on validation Macro-F1
        |
        v
Restore best model weights
        |
        v
Save training history and learning curves
        |
        v
Log parameters, metrics, and artifacts to MLflow
        |
        v
Repeat for every candidate configuration
        |
        v
Compare validation Macro-F1
        |
        v
Inspect learning curves and overfitting
        |
        v
Select final configuration
        |
        v
Evaluate selected model once on test set
```

---

## 11. Model-selection criterion

The final configuration is selected primarily according to validation Macro-F1.

However, validation Macro-F1 is interpreted together with the training dynamics.

A candidate is preferred when it:

1. achieves strong validation Macro-F1;
2. performs better than the baseline or provides a meaningful trade-off;
3. does not exhibit clear and unnecessary overfitting;
4. does not rely on substantially greater model complexity without corresponding validation improvement.

If two configurations have very similar validation Macro-F1, the simpler configuration is preferred unless the more complex configuration provides a clear practical advantage.

The test result is not considered when choosing between candidate configurations.

---

## 12. Final test evaluation

After model selection has been completed, the **selected** model is evaluated on the previously unseen test partition.

The same prediction processing used during validation is applied:

```text
Test sequences
      |
      v
Selected model
      |
      v
Softmax probabilities
      |
      v
Boundary trimming
      |
      v
Overlapping prediction aggregation
      |
      v
One prediction per retained original timestep
      |
      v
Final test metrics
```

The final test evaluation reports at least:

- Accuracy;
- Macro-F1;
- confusion matrix;
- number of evaluated predictions.

The resulting test performance represents the performance of the **selected** model on an unseen partition and is not subsequently used to modify the model-selection decision.

---

## 13. Reproducibility

Every model-selection run should be identifiable by its configuration and associated MLflow run.

The following should remain fixed across candidate experiments unless explicitly being investigated:

- model-dataset version;
- train/validation/test split;
- preprocessing procedure;
- sequence-generation procedure;
- evaluation procedure;
- class definitions.

Random seeds should be controlled where reproducibility is required.

Exact numerical reproducibility may nevertheless depend on the TensorFlow version, hardware, operating system, and execution environment.

---

## 14. Scope of the initial experiment

The initial experiment deliberately focuses on a small set of LSTM configurations.

More extensive hyperparameter optimization is not performed unless the initial controlled comparison indicates that further optimization is justified.

If the LSTM candidates do not provide satisfactory performance, alternative sequence architectures may subsequently be investigated, such as:

- GRU;
- Temporal Convolutional Networks (TCNs);
- Transformer-based sequence models.

Such models should be evaluated under the same data and timestep-level evaluation contract so that their results remain comparable with the LSTM experiments.

---

# Results and findings

## 15. Candidate configurations

The initial model-selection experiment consisted of a small number of deliberately chosen configurations. All formal candidate runs used the same data split, sequence-processing procedure, training settings, and random seed (`42`), with only the specified hyperparameter or architectural factor changed.

| Configuration | LSTM units | Dropout | Learning rate | Purpose |
|---|---|---:|---:|---|
| A | `(64,)` | 0.30 | `1e-3` | Baseline |
| B | `(64,)` | 0.50 | `1e-3` | Investigate stronger regularization |
| C | `(128,)` | 0.30 | `1e-3` | Investigate increased model capacity |
| D | `(64, 32)` | 0.30 | `1e-3` | Investigate increased depth |
| E | `(64,)` | 0.30 | `5e-4` | Investigate a lower learning rate |

For all configurations:

- sequence length = 7;
- number of input features = 41;
- number of classes = 4;
- boundary width = 2;
- maximum epochs = 50;
- batch size = 256;
- early-stopping patience = 5;
- test evaluation = disabled (`evaluate_test=False`);
- random seed = 42.

The test set was therefore kept isolated throughout the model-selection stage.

### 15.1 Selection criterion

The primary model-selection metric is validation Macro-F1. This is consistent with the class-imbalanced nature of the task and avoids allowing the dominant class to determine the model-selection outcome.

Early stopping monitors validation Macro-F1. After training, the restored best model is evaluated using the same sequence-boundary trimming and overlapping-window aggregation procedure used for the final validation evaluation.

For the baseline experiment, the best validation Macro-F1 from the training history exactly matched the post-training validation Macro-F1, confirming consistency between the early-stopping metric and the final validation evaluation.

### 15.2 Candidate results

The controlled A–E experiments produced the following results:

| Rank | Configuration | Best validation Macro-F1 | Best epoch | Validation accuracy |
|---:|---|---:|---:|---:|
| 1 | **B -- `(64,)`, dropout 0.50** | **0.851805** | 6 | **0.8933** |
| 2 | A -- `(64,)`, dropout 0.30 | 0.851387 | 7 | 0.8929 |
| 3 | D -- `(64, 32)`, dropout 0.30 | 0.850658 | 2 | 0.8914 |
| 4 | E -- `(64,)`, dropout 0.30, LR `5e-4` | 0.850559 | 14 | 0.8924 |
| 5 | C -- `(128,)`, dropout 0.30 | 0.849866 | 3 | 0.8917 |

The stronger-dropout configuration B achieved the highest validation Macro-F1, but its improvement over the baseline A was only approximately 0.00042 Macro-F1.

Increasing the number of units from 64 to 128 (C) did not improve validation performance. Similarly, increasing model depth (D) and reducing the learning rate (E) did not produce an improvement over the baseline.

### 15.3 Learning-curve observations

The learning curves indicate rapid convergence for the candidate models, followed by validation-performance saturation or mild decline.

Configuration A reached its best validation Macro-F1 at epoch 7. Configuration B reached its best at epoch 6. Configuration C peaked much earlier, at epoch 3, while D peaked at epoch 2. Configuration E required substantially more epochs and reached its best validation Macro-F1 at epoch 14.

In general, training Macro-F1 continued to increase after the best validation epoch, while validation Macro-F1 fluctuated or declined. This indicates mild generalization saturation rather than severe overfitting.

The results do not provide evidence that increasing model capacity is beneficial for this task.

### 15.4 Random-seed robustness check

Because the difference between configurations A and B under seed 42 was very small, an additional robustness check was performed for the baseline configuration A using seeds 43 and 44.

| Run | Configuration | Seed | Best validation Macro-F1 | Best epoch |
|---|---|---:|---:|---:|
| A | `(64,)`, dropout 0.30 | 42 | 0.851387 | 7 |
| A43 | `(64,)`, dropout 0.30 | 43 | 0.8511 | 4 |
| A44 | `(64,)`, dropout 0.30 | 44 | 0.8512 | 11 |

The baseline therefore produced validation Macro-F1 values around 0.851 across all three tested seeds, indicating that its performance is reasonably stable with respect to random initialization and stochastic training.

The stronger-dropout configuration B has so far been evaluated under seed 42 only. Therefore, its approximately 0.00042 improvement over A cannot yet be regarded as a robust advantage.

### 15.5 Current model-selection conclusion

The controlled candidate search does not indicate a meaningful benefit from increasing LSTM capacity, adding depth, or reducing the learning rate. Configuration B achieved the highest validation Macro-F1 under seed 42, but its improvement over the baseline A was very small.

Because this difference could be attributable to stochastic variation, the two strongest candidates, A and B, were subsequently evaluated under three random seeds: 42, 43, and 44.

Configuration A produced validation Macro-F1 values consistently around 0.851 across the three seeds. Configuration B achieved 0.851805 under seed 42, but its performance decreased to approximately 0.8506 under seeds 43 and 44. Thus, the apparent advantage of B under seed 42 was not reproduced consistently.

The multi-seed results therefore favor the simpler baseline configuration A as the more robust choice:

- **LSTM architecture:** `(64,)`
- **Dropout:** `0.30`
- **Learning rate:** `1e-3`
- **Batch size:** `256`
- **Maximum epochs:** `50`
- **Early-stopping patience:** `5`

Configuration A is consequently selected for the final evaluation.

Importantly, the test set has not been used during this model-selection process. All candidate configurations and random-seed checks were evaluated exclusively using the training and validation data.

### 15.6 Final model-selection rationale

The final selection is based on both validation performance and robustness rather than on the single highest observed validation score.

Although Configuration B obtained the highest individual validation Macro-F1 (0.851805), its advantage over A was approximately 0.0004 and disappeared under the additional random seeds. Configuration A showed stable performance across seeds while also using the simpler architecture and lower dropout rate.

The selection therefore prioritizes:

1. validation Macro-F1 as the primary performance criterion;
2. consistency across random seeds;
3. simplicity of the model architecture;
4. avoidance of unnecessary model capacity or regularization;
5. complete isolation of the test set during model selection.

This procedure avoids selecting a configuration solely because it happened to obtain the highest score under one stochastic training realization.

### 15.7 Final configuration

The selected configuration for final evaluation is:

| Parameter | Selected value |
|---|---:|
| LSTM units | `(64,)` |
| Dropout rate | `0.30` |
| Learning rate | `1e-3` |
| Sequence length | `7` |
| Number of features | `41` |
| Number of classes | `4` |
| Boundary width | `2` |
| Batch size | `256` |
| Maximum epochs | `50` |
| Early-stopping patience | `5` |
| Test evaluation during selection | `False` |

The final model will be trained using the established training procedure and evaluated on the held-out test set only after model selection has been completed.

### 15.8 Test-set evaluation

The held-out test set remains untouched throughout model selection. In particular, `evaluate_test=False` was used for all candidate configurations and all random-seed robustness checks.

After selecting Configuration A, a final experiment will be performed with the same model configuration and `evaluate_test=True`. The resulting test metrics will provide the final unbiased estimate of performance on unseen data.

The test evaluation will use exactly the same sequence-level evaluation procedure as the validation evaluation:

1. discard predictions near sequence boundaries;
2. aggregate overlapping sequence predictions referring to the same original timestep;
3. average the corresponding class-probability vectors;
4. determine the final predicted class using `argmax`;
5. match predictions with the corresponding ground-truth labels;
6. calculate timestep-level accuracy and Macro-F1;
7. construct the confusion matrix.

No model-selection decision will be made using these test results.

### 15.9 MLflow experiment tracking

Each model-selection run is tracked as a separate MLflow run within the dedicated model-selection experiment.

The logged parameters include the model architecture and training configuration, including:

- LSTM-unit configuration;
- dropout rate;
- learning rate;
- sequence length;
- number of features;
- number of classes;
- boundary width;
- batch size;
- maximum epochs;
- early-stopping patience;
- `evaluate_test`;
- random seed.

The main validation results are logged as metrics, including validation accuracy, validation Macro-F1, and the best validation Macro-F1. The best epoch and model-selection stage are recorded as tags.

The experiment uses explicit MLflow logging rather than MLflow autologging. This provides control over exactly which parameters, metrics, tags, and artifacts constitute the experiment record and avoids automatically logging framework-generated information that is not required for the project's reproducibility or model-selection analysis.

The dedicated tracking functionality is implemented separately from the model-training logic so that experiment execution and experiment bookkeeping remain distinct.

### 15.10 Reproducibility

Random seeds are explicitly recorded as experiment parameters. This makes it possible to distinguish differences caused by hyperparameter changes from differences arising from stochastic training.

The model-selection experiments also persist the training history and learning-curve artifacts. The numerical training history is saved as CSV, and the corresponding learning curves are generated from the persisted CSV rather than directly from the in-memory Keras `History` object.

This makes the saved training-history CSV the numerical source of truth for the learning-curve artifacts.

### 15.8 Final held-out test evaluation

After completion of model selection, Configuration A was selected for final evaluation:

- LSTM units: `(64,)`
- Dropout rate: `0.30`
- Learning rate: `1e-3`
- Random seed: `42`
- `evaluate_test=True`

The final model was trained using the established training procedure with early stopping based on validation Macro-F1. The best validation performance occurred at epoch 7.

The final validation and test results were:

| Dataset | Accuracy | Macro-F1 | Number of predictions |
|---|---:|---:|---:|
| Validation | 0.8929 | 0.851387 | 10,000 |
| **Test** | **0.8920** | **0.850308** | **10,000** |

The final test Macro-F1 of **0.8503** is close to the validation Macro-F1 of **0.8514**, with a difference of approximately 0.0011. This indicates that the selected model generalizes to the held-out test data without a substantial degradation relative to its validation performance.

The test confusion matrix was:

\[
\begin{pmatrix}
5483 & 221 & 0 & 0 \\
292 & 1438 & 157 & 0 \\
0 & 152 & 1250 & 103 \\
0 & 0 & 155 & 749
\end{pmatrix}
\]

where rows correspond to the true classes and columns to the predicted classes.

The largest off-diagonal errors occur between neighboring classes, particularly between classes 0 and 1, classes 1 and 2, and classes 2 and 3. This indicates that the remaining classification errors are concentrated in transitions between operational states rather than being uniformly distributed across unrelated classes.

The final MLflow run was recorded under the `final_evaluation` selection stage. The test set was not used for any hyperparameter, architecture, or model-selection decision.

### 15.9 Final model-selection outcome

The complete model-development procedure can therefore be summarized as follows:

1. A baseline LSTM configuration was established.
2. Four alternative configurations were evaluated to investigate dropout, model capacity, model depth, and learning rate.
3. Validation Macro-F1 was used as the primary model-selection criterion.
4. The test set was excluded from all candidate-model experiments.
5. Configurations A and B were subjected to additional random-seed robustness checks because their initial performance difference was very small.
6. Configuration A demonstrated more consistent performance across seeds and was selected as the final configuration.
7. The selected configuration was then evaluated once on the previously untouched test set.
8. The final test performance was **0.8920 accuracy** and **0.8503 Macro-F1**.

The final result provides an unbiased estimate of the selected model's performance on unseen data under the established evaluation procedure.





---
---
---

# Appendix


## MLflow tracking design

MLflow is used during model selection to provide reproducible tracking and comparison of candidate model configurations. Each candidate configuration is recorded as a separate MLflow run within the shared LSTM model-selection experiment.

### Explicit tracking instead of MLflow autologging

The project uses explicit MLflow logging rather than `mlflow.autolog()`. The primary reason is that model selection relies on project-specific evaluation procedures and metrics, particularly the aggregated validation Macro-F1 obtained after sequence-boundary trimming and overlapping-prediction aggregation. Explicit logging provides control over which parameters, metrics, and artifacts constitute the experiment record and avoids relying on automatically inferred framework-specific metrics.

### Separation of experiment execution and tracking

MLflow-specific functionality is isolated in `src/finance_ml/models/tracking.py`. The experiment orchestration code is responsible for building, training, and evaluating the model, whereas the tracking module is responsible for recording the resulting parameters, metrics, and artifacts in MLflow. This separation keeps the core modeling code independent of the tracking infrastructure and allows the model code to remain testable in environments where MLflow is not required.

### MLflow run lifecycle

The caller controls the MLflow run lifecycle using `mlflow.start_run()`. The tracking utilities do not create or terminate runs themselves; they only log information to an already active run. This makes the boundary between experiment execution and experiment tracking explicit and prevents tracking utilities from unexpectedly creating additional runs.

### Model-selection and test-set separation

During model selection, candidate configurations are evaluated using the training and validation sets only. Validation Macro-F1 is the primary model-selection metric. Test-set evaluation is disabled for candidate runs and is performed only after the final configuration has been selected. Consequently, test metrics are not used to select, tune, or compare candidate configurations.


```Python
              LSTM experiment
                      │
     ┌────────────────┼────────────────┐
     │                │                │
configuration      results           files
     │                │                │
     ▼                ▼                ▼
 Parameters        Metrics         Artifacts
```



```Python
The MLFlow Schema for each logged model:

LSTM_Model
│
├── Parameters
│   ├── sequence_length
│   ├── n_features
│   ├── n_classes
│   ├── lstm_units
│   ├── dropout_rate
│   ├── learning_rate
│   ├── boundary_width
│   ├── epochs
│   ├── batch_size
│   ├── patience
│   └── evaluate_test
│
├── Metrics
│   ├── validation_accuracy
│   ├── validation_macro_f1
│   └── best_val_macro_f1
│
├── Tags (metadata)
│   ├── model_type
│   ├── selection_stage
│   └── best_epoch
│
└── Artifacts
    ├── training_history.csv
    ├── learning_curve_loss.png
    ├── learning_curve_accuracy.png
    └── learning_curve_macro_f1.png
```

```Python
08_model_selection_mlflow.ipynb
│
├── 1. Project/import/environment setup
├── 2. Load prepared model dataset (Gold data)
├── 3. prepare_model_dataset()
├── 4. Verify dataset
├── 5. MLflow experiment setup
│
├── 6. Define Configuration A
├── 7. Start MLflow run
├── 8. run_lstm_experiment
│       └── (evaluate_test=False)
├── 9. log_sequence_experiment(...)
├── 10. Inspect Run A in Databricks
│
├── Configuration B
├── Configuration C
├── ...
│
└── Compare MLflow runs
        ↓
   Select winner
        ↓
   final test evaluation
```