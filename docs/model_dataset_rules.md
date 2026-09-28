# Model Dataset Rules

The model dataset is built from:

    finance_ml.gold.account_features

Its purpose is to prepare leakage-safe train, validation, and test data for
financial time-series classification.

## Target

The supervised target is:

- `risk_state`

The human-readable label is:

- `risk_state_name`

`risk_state_name` may be retained for reporting, but it must not be used as a
model input feature.

## Identifier columns

The following columns identify observations but are not model input features:

- `account_id`
- `date`

They are retained for ordering, splitting, evaluation, and traceability.

## Candidate model features

The initial feature set includes cleaned Gold predictors such as:

- `balance`
- `cash_inflow`
- `cash_outflow`
- `transaction_count`
- `credit_utilization`
- `amount_due`
- `amount_paid`
- `payment_ratio`
- `days_past_due`
- `net_cash_flow`
- `payment_gap`
- lag features
- change features
- 7-day rolling features

Categorical variables such as `account_type` require explicit encoding before
being passed to a numerical model.

## Leakage prevention

Current and future target values must never be included in the input feature
matrix.

Train, validation, and test splitting must respect time ordering.

Random row-level splitting is not allowed because it can place future
observations from the same account into the training set while earlier
observations from that account appear in validation or test data.

## Temporal splitting

The initial split will use chronological date boundaries.

All observations before the first boundary belong to training.

Observations between the first and second boundaries belong to validation.

Observations after the second boundary belong to test.

Exact boundaries will be chosen after inspecting the Gold date range.

## Missing values

Remaining missing values must be handled explicitly before model training.

Missingness must not be filled using future observations.

The handling strategy may differ by feature type.

Missing values should not be silently replaced without documenting the rule.

## Sequence preparation

For sequence models, observations must be:

- grouped by `account_id`
- ordered by `date`

Sequence windows must never cross account boundaries.

Sequence construction must also preserve temporal causality.

The initial sequence length will be selected later based on the modeling
experiment design.

## Class imbalance

The four risk states are not equally frequent.

Evaluation should therefore include metrics such as:

- macro F1
- per-class precision
- per-class recall
- confusion matrix

Accuracy alone is not sufficient.


## Numeric missing-value strategy

Remaining numeric missing values are handled during model-dataset preparation.

For each numeric feature containing missing values:

1. A binary missing-indicator feature is created.
2. The numeric missing values are replaced using the median calculated from the
   training split only.
3. The same training-derived median is applied to validation and test data.

Validation and test data must never contribute to imputation statistics; because otherwise information from the future evaluation sets leaks into preprocessing.

Missing indicators preserve the distinction between an observed value and an
imputed value.

This is particularly important for variables where zero has a real meaning,
such as cash flows and change features. Missing values must therefore not be
silently replaced with zero.

## Categorical encoding

Categorical model features are encoded using category levels learned from the
training split only.

For the initial dataset, `account_type` is one-hot encoded.

The training-derived category levels define the feature schema used for
training, validation, and test data.

Validation and test data must not introduce new model feature columns.

A category that was not observed during training (but exists in validation or 
test split) is represented by zeros across the training-defined one-hot columns.



## Target representation

The initial sequence-classification model uses integer class labels rather than
one-hot encoded targets.

The four risk states are represented as:

- `0`: normal
- `1`: elevated risk
- `2`: liquidity stress
- `3`: delinquency risk

For sequence-to-sequence classification, the target array therefore has shape:

    (n_sequences, sequence_length)

rather than:

    (n_sequences, sequence_length, n_classes)

The model output still contains one probability distribution over the four
classes at each timestep, for example through:

    Dense(4, activation="softmax")

Because the targets remain integer encoded, the corresponding loss function is:

    sparse_categorical_crossentropy

If the targets were one-hot encoded instead, the appropriate loss would be:

    categorical_crossentropy

The sparse representation is preferred here because it is simpler and avoids
unnecessarily expanding the target array.



## Why input categories are one-hot encoded but targets are not

Categorical input features and categorical targets play different roles in the
model and therefore do not require the same representation.

For an input feature such as `account_type`, categories such as:

- `business`
- `checking`
- `credit`

are nominal and have no natural numerical ordering.

Encoding them as integers such as:

    business = 0
    checking = 1
    credit = 2

would expose those numbers directly to the model and could introduce an
artificial ordering or distance between categories.

For that reason, categorical input features are one-hot encoded, for example:

    business -> [1, 0, 0]
    checking -> [0, 1, 0]
    credit   -> [0, 0, 1]

The target `risk_state` is different.

Its integer values:

    0, 1, 2, 3

are not treated as ordinary numerical input values. When the model is trained
with `sparse_categorical_crossentropy`, these integers are interpreted only as
class indices.

The loss function therefore understands:

    0 -> class 0
    1 -> class 1
    2 -> class 2
    3 -> class 3

and does not interpret class 3 as being numerically three times class 1.

The model output still contains a probability distribution over all classes at
each timestep:

    [P(class 0), P(class 1), P(class 2), P(class 3)]

Therefore:

- categorical input features are one-hot encoded to avoid introducing false
  numerical structure;
- categorical targets can remain integer encoded when using
  `sparse_categorical_crossentropy`.

If the targets were one-hot encoded instead, the corresponding loss would be
`categorical_crossentropy`.



## Feature scaling

Continuous numeric model features are scaled using statistics calculated from
the training split only.

The initial project supports two scaling strategies.

### Standard scaling

Standard scaling uses the training mean and training standard deviation:

    scaled_value =
        (value - training_mean) / training_standard_deviation

This method is simple and widely used, but the mean and standard deviation can
be strongly influenced by extreme values.

### MAD-based robust scaling

For financial variables that may contain heavy tails or large legitimate
spikes, a more robust alternative is based on the median and median absolute
deviation (MAD).

For each feature:

    median =
        median(x)

    MAD =
        median(|x - median(x)|)

The MAD is multiplied by the Gaussian consistency factor:

    corrected_MAD =
        1.4826 * MAD

and the scaled value is:

    scaled_value =
        (value - training_median) / corrected_MAD

The factor `1.4826` makes the MAD-based scale approximately comparable to the
standard deviation when the underlying distribution is Gaussian.

MAD-based scaling is less sensitive to outliers than mean/standard-deviation
scaling and is therefore a useful default candidate for financial features such
as cash flows, balance changes, and other potentially heavy-tailed variables.

All centering and scaling parameters must be estimated from the training split
only. The same training-derived parameters are then applied to training,
validation, and test data.

Validation and test data must never contribute to the scaling statistics.

If a feature has zero MAD, the scaler must use a safe fallback so that division
by zero does not occur. A possible strategy is to fall back to the training
standard deviation when it is non-zero, and otherwise use a scale of `1.0` for
a constant feature.

Binary missing-indicator features and one-hot encoded categorical features are
not scaled because they already have a meaningful 0/1 representation.

The project should retain standard scaling as a baseline and allow MAD-based
robust scaling to be selected explicitly so that both approaches can be
compared empirically.



## Sequence-modeling design

The initial deep-learning formulation uses sequence-to-sequence classification.

For each account, a sequence of model-ready feature vectors is constructed in
chronological order:

    x_1, x_2, ..., x_L

The corresponding target sequence is:

    y_1, y_2, ..., y_L

where each `y_t` is the `risk_state` associated with timestep `t`.

The sequence builder must:

- group rows by `account_id`
- order rows by `date`
- never mix rows from different accounts
- preserve chronological order
- never use future rows to construct earlier timesteps
- preserve one target label per input timestep

Sequence windows must be created separately inside the train, validation, and
test partitions.

A sequence must never cross a train/validation/test boundary.

The initial sequence length will be configurable rather than hard-coded.



## Initial sequence configuration

The initial sequence-to-sequence dataset uses:

- sequence length: 7 timesteps
- stride: 1 timestep
- final numerical feature count: 41

Sequences are created independently within each account and separately within
the train, validation, and test partitions.

The resulting tensor shapes are:

    X_train      = (72,000, 7, 41)
    y_train      = (72,000, 7)

    X_validation = (6,000, 7, 41)
    y_validation = (6,000, 7)

    X_test       = (6,000, 7, 41)
    y_test       = (6,000, 7)

The target tensors contain integer class indices and are intended for use with
`sparse_categorical_crossentropy`.



## Overlapping sequence predictions and boundary handling

Sequences are generated with `stride = 1`, so consecutive sequence windows
overlap strongly.

For example, with `sequence_length = 7` and `stride = 1`, one account may
produce consecutive windows such as:

    Window 1: t1, t2, t3, t4, t5, t6, t7
    Window 2:     t2, t3, t4, t5, t6, t7, t8
    Window 3:         t3, t4, t5, t6, t7, t8, t9

So neighboring windows share most of their timesteps.

For instance, timestep `t5` appears in all three windows above, but at different
positions inside each window. This allows the same original timestep to be
classified using several overlapping temporal contexts.

This overlap is intentional. A timestep may therefore receive predictions from
multiple sequence windows.

Predictions near the beginning or end of a sequence can be less reliable than
predictions near the center because boundary positions have less within-window
temporal context.

For a sequence of length 7:

    positions = 0, 1, 2, 3, 4, 5, 6

the initial prediction-aggregation strategy uses a boundary width of 2 and
retains only the central positions:

    retained positions = 2, 3, 4

Predictions produced at positions 0, 1, 5, and 6 are discarded for final
timestep-level evaluation.

Because overlapping windows can still produce multiple retained predictions
for the same `(account_id, date)` timestep, their predicted class-probability
vectors are averaged.

The final predicted class is then obtained from the averaged probability
vector using `argmax`.

This strategy:

- reduces sensitivity to sequence-boundary effects;
- uses overlapping windows to provide multiple contextualized predictions;
- produces one final prediction per original timestep;
- preserves chronological and account boundaries.

The boundary width should remain configurable and may later be compared
empirically against alternatives such as:

- keeping only the exact center timestep;
- averaging all overlapping predictions;
- weighting predictions by distance from the sequence center.