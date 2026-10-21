# Sequence Prediction Policies: Training, Validation, and API Serving

**Project:** Financial Time-Series ML Platform  
**Suggested repository location:** `docs/models/sequence_prediction_policies.md`  
**Scope:** Current unidirectional seq2seq LSTM, not every possible sequence model.

## 1. Overview

Our LSTM classifies the financial risk state of each account-day. The model takes a window of `sequence_length` observations and returns a probability vector for **every** position in that window.

For the present checkpoints, `sequence_length = 7`, `n_features = 41`, and `n_classes = 4`:

```text
Input X:  (batch, 7 timesteps, 41 features)
                     |
                     v
             Unidirectional LSTM
                     |
                     v
           TimeDistributed Softmax
                     |
                     v
Output P: (batch, 7 timesteps, 4 classes)
```

These *three policies* specify **which outputs we use and why** at three different stages. They do **not** represent three separately trained models.

```text
Positions:       1   2   3   4   5   6   7

Training loss:   ✓   ✓   ✓   ✓   ✓   ✓   ✓

Validation:      ×   ×   ✓   ✓   ✓   ×   ×

API inference:   ×   ×   ×   ×   ×   ×   ✓
```

Here, `✓` means that the policy uses the prediction at that position, not that the predicted class is correct. The validation pattern assumes `boundary_width = 2`; both lengths are configurable for training.

## 2. Training policy — loss at every position

**Implementation:** `src/finance_ml/models/lstm.py`, `src/finance_ml/models/training.py`.

The seq2seq model produces a risk-class distribution at each of the seven timesteps. Training uses `sparse_categorical_crossentropy` against the label at **every** position:

```text
Day:         D1   D2   D3   D4   D5   D6   D7
Input:       x1   x2   x3   x4   x5   x6   x7
Output:      p1   p2   p3   p4   p5   p6   p7
Target:      y1   y2   y3   y4   y5   y6   y7
Loss:        ✓    ✓    ✓    ✓    ✓    ✓    ✓
```
Note: each output (p1 to p7) is a probability vector, containing 4 class-probabilities.

**Why:** Every timestep provides supervised learning signal, rather than training on only one label per window. This is natural for a sequence-to-sequence classifier and efficiently uses available labels. Overlapping windows can still weight an original date multiple times in the loss; they are distinct training examples with different within-window context.

**Important:** Keras `accuracy` and `val_accuracy` in the standard training logs are calculated over the model's timestep outputs. They are not the same metric as our custom aggregated `macro_f1` / `val_macro_f1` callbacks.

## 3. Validation policy — trim boundaries and aggregate overlaps

**Implementation:** `src/finance_ml/models/training.py`, `src/finance_ml/models/evaluation.py`, `src/finance_ml/models/prediction_aggregation.py`.

After each epoch, our custom callback predicts on the validation sequences. For `sequence_length = 7` and `boundary_width = 2`, it discards positions 1–2 and 6–7 of **each window** and retains positions 3–5.

```text
Window A:       D1  D2 [D3  D4  D5] D6  D7
Window B:       D2  D3 [D4  D5  D6] D7  D8
Window C:       D3  D4 [D5  D6  D7] D8  D9
                           ^
             Same original day can contribute
             probabilities from several windows.
```

The evaluation groups retained predictions by `(account_id, date)`, **averages their class-probability vectors**, chooses a class with `argmax`, and computes macro-F1 against **one true label per original account/date**.

**Why:**

1. Trimming was chosen to avoid relying on sequence-boundary outputs in the aggregated validation metric.
2. Aggregation prevents a date appearing in several overlapping windows from being counted as several separate evaluation observations.
3. Macro-F1 weights all four risk classes equally, which is useful when class frequencies are imbalanced.
4. `val_macro_f1` governs checkpoint selection, learning-rate reduction (when enabled), and early stopping. Thus the reported *best* validation macro-F1 refers to this specific trimming-and-aggregation policy.

**Nuance:** Our LSTM is **unidirectional**. At position 3, it has processed only positions 1–3 of that particular window, not all seven positions; at position 5, it has processed positions 1–5. It does **not** see later observations within the window. Symmetric trimming is an evaluation design choice, not proof that both window edges are equally problematic or that the retained outputs have seven steps of context.

## 4. API serving policy — causal endpoint prediction

**Implementation:** `src/finance_ml/pipelines/inference_runner.py` (`causal_endpoint_predictions`) and `src/finance_ml/api/prediction_pipeline.py`.

To classify account A on a requested date `T`, the API takes the latest `sequence_length` available observations **ending at T**, runs the saved LSTM, and selects only the probability vector at the last position:

```text
Request: risk state for D7

Input window:   D1   D2   D3   D4   D5   D6   D7
Model output:   p1   p2   p3   p4   p5   p6   p7
                                            ^
                                        use p7
```

The key code is:

```python
probabilities = predictions[:, -1, :]
```

**Why:** It returns a prediction for the **requested date** based on current and prior observations, without requiring observations *after* that date. The saved model and training-fitted imputation, encoding, scaling and feature order are reused; the API does not retrain or refit anything.

For a length-7 model, the first six observed endpoints for a new account have `insufficient_context` status; the seventh and later endpoints can be eligible. The requirement is derived from saved model metadata rather than being hardcoded. A seven-row window is not necessarily seven consecutive calendar days: date gaps are reported separately under the current policy. Raw-data feature engineering can need extra context for complete lag and rolling features.

**Interpretation:** This is a **same-date classification**, not a prediction of an unobserved future day. The complete preprocessing chain must also be checked for unintended look-ahead.

## 5. Why validation macro-F1 is not the API's macro-F1

Both stages use **the same trained network weights**, but different output-selection policies:

| Property | Validation metric | API inference |
|---|---|---|
| Sequence outputs selected | Middle positions 3–5 | Last position 7 |
| Multiple predictions for one date | Average retained probability vectors | One trailing-window endpoint prediction |
| Available context per retained output | 3–5 within-window steps in this forward LSTM | 7 within-window steps |
| Ground-truth labels needed | Yes, for macro-F1 | No, for prediction |
| Typical output | Validation macro-F1, confusion matrix | Predicted risk class, probabilities, coverage |

Consequently, **a validation macro-F1 of 0.8611 is not a measured API-serving macro-F1**. A reliable serving-performance estimate requires a labeled evaluation dataset scored using the exact causal endpoint policy.

### Recommended additional evaluation

```text
Common held-out account-day dataset
                 |
                 v
    Apply each model's saved preprocessing
                 |
                 v
       Trailing sequence windows
                 |
                 v
      Keep final-timestep predictions
                 |
                 v
   Compare to independently held-out labels
                 |
                 v
   Macro-F1 / per-class F1 / confusion matrix
```

Use the **same evaluation account-dates** for candidate models, do not use test labels to tune hyperparameters, and preserve the exact version and preprocessing metadata of each candidate. The API's coverage policy and any unavailable endpoints should also be reported.

## 6. Comparing the two local model experiments

The prior lightweight model reportedly achieved about **0.74 validation macro-F1**, while the stronger run achieved **0.861084** with its best epoch at epoch 4.

If the earlier value was computed with the same `sequence_length = 7`, `boundary_width = 2`, and trimming/aggregation code, then **the two numbers use a comparable metric definition**. It is reasonable to say that the new experiment reported a higher validation macro-F1 **on its own validation dataset**.

But they are **not yet an apples-to-apples performance comparison**:

- The earlier run used about 20 accounts × 60 days; the new run used 60 accounts × 90 days.
- The temporal training/validation partitions and evaluated account-dates differ.
- Larger or different simulation cohorts can change class balance and task difficulty, even with the same simulator seed.
- Both values measure *trimmed-and-aggregated validation*, not causal endpoint serving.

Therefore, do not claim that a larger architecture itself caused the improvement, or that API performance rose by the same amount. Run both saved checkpoints on a **common independent evaluation dataset**, ideally reporting both the original validation policy and the real endpoint-serving policy.

## 7. Future options

We can preserve the current model while adding a **serving-aligned validation metric** based on last-position predictions. Later experiments could monitor that metric for checkpoint selection, if endpoint inference is the intended production use case. Such a change would alter the experiment protocol and should be documented and versioned rather than silently replacing historical validation results.
