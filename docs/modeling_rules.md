1. Overview and modeling objective
2. Sequence formulation (seq2seq classification)
3. Input representation and target encoding
4. LSTM architecture design
5. Configurable stacked LSTM architecture
6. Training strategy and early stopping
7. Validation macro-F1 computation
8. Overlapping sequence windows and prediction aggregation
9. Boundary trimming strategy
10. Final evaluation procedure
11. Experiment orchestration workflow
12. Reproducibility and future model extensions


# Modeling Rules and Experiment Design

## 1. Overview and modeling objective

This document describes the design principles, assumptions, and evaluation
procedures used for the sequence-classification models in the project.

The objective is to classify the operational state of each account at each
timestep based on a sequence of historical observations. Therefore, the
problem is formulated as a **many-to-many sequence classification problem**
(seq2seq classification), where each input sequence produces one prediction
for every timestep within that sequence.

The model receives a fixed-length historical window:

\[
X_t =
(x_{t}, x_{t+1}, ..., x_{t+L-1})
\]

where:

- \(L\) is the sequence length,
- each timestep contains a vector of engineered features,
- the output is a sequence of class predictions:

\[
\hat{Y}_t =
(\hat{y}_{t}, \hat{y}_{t+1}, ..., \hat{y}_{t+L-1})
\]

The model is not designed to predict only the final timestep of a window.
Instead, it learns temporal dependencies and produces a classification
decision at every timestep.

This formulation allows the model to exploit temporal patterns while
maintaining alignment between predictions and the original time-series
records.

## 2. Sequence formulation: seq2seq classification

The model-dataset construction produces fixed-length sequences of seven
consecutive timesteps. For each sequence, the model receives a feature matrix
with shape

    (7, 41)

and produces one class prediction for each of the seven timesteps.

The task is therefore formulated as **sequence-to-sequence (seq2seq)
classification**, rather than sequence-to-one classification.

For an input sequence

    X = (x1, x2, ..., x7)

the corresponding target is

    Y = (y1, y2, ..., y7)

where each yt is an integer representing one of the four target classes.

The model consequently produces

    Ŷ = (ŷ1, ŷ2, ..., ŷ7)

with one predicted class-probability vector for every timestep.

The model is not designed to predict only the final timestep of a window.
Instead, it learns temporal dependencies and produces a classification
decision at every timestep.

### Why seq2seq is used

A sequence-to-one formulation would use the entire seven-timestep window to
produce only one target, typically corresponding to the final timestep. That
would discard the classification targets associated with the other timesteps
and would not match the structure of the prepared model dataset.

The seq2seq formulation instead preserves the temporal alignment between
input observations and target labels. Each timestep can therefore contribute
a classification output while the recurrent layers use information from the
surrounding sequence to learn temporal dependencies.

This formulation is particularly useful for the present overlapping-window
strategy because the same original timestep can occur in multiple consecutive
sequence windows. Its multiple predictions can subsequently be combined
during the prediction-aggregation stage.

### Sequence length and feature dimension

The current model-dataset configuration uses:

- **Sequence length:** 7 timesteps
- **Features per timestep:** 41
- **Target classes:** 4

Thus:

    X: (batch, 7, 41)
    Y: (batch, 7)
    ŷ: (batch, 7, 4)

The final dimension of the model output contains the four class probabilities
produced by the softmax classification head.

The target does not contain a class-probability vector. It remains an integer
class label at each timestep. Consequently, sparse multiclass
cross-entropy is used as the training loss.

## 3. Input representation and target encoding

The model receives the prepared model dataset after the preceding data
processing stages have been completed. The input representation and target
encoding are deliberately treated differently because they serve different
purposes.

### Input features

The model input contains 41 features per timestep. These features consist of
the numerical and encoded categorical variables selected during model-dataset
preparation.

Categorical input variables are represented using one-hot encoding where
appropriate. For example, the account-type categories are transformed into
separate binary feature columns:

    account_type_business
    account_type_checking
    account_type_credit

This encoding allows the neural network to process categorical information
numerically without imposing an artificial ordering between categories.

Continuous numerical features are standardized using statistics estimated
from the training partition only. Validation and test observations are
transformed using those already-fitted training statistics.

This separation is essential to prevent information from the validation or
test partitions from influencing the feature transformation applied to the
training data.

### Target representation

The target variable represents one of four mutually exclusive classes:

    0 = normal
    1 = elevated_risk
    2 = liquidity_stress
    3 = delinquency_risk

The targets are intentionally kept as integer class labels rather than
one-hot encoded vectors.

For a sequence of seven timesteps, the target therefore has the form:

    [y1, y2, y3, y4, y5, y6, y7]

where each yt is an integer in the range 0--3.

The corresponding model output has four probabilities for each timestep:

    [
        [p(class 0), p(class 1), p(class 2), p(class 3)],
        ...
    ]

Sparse integer targets are used with
`sparse_categorical_crossentropy`. One-hot encoding is unnecessary because
the loss function directly accepts integer class indices.

This distinction is intentional:

- Input categorical variables are one-hot encoded because their category
  membership must be represented as numerical input features.
- Target labels remain integer class IDs because the multiclass loss function
  directly operates on those class IDs.

Keeping the target sparse also avoids introducing unnecessary dimensionality
into the target representation while preserving exactly the same class
information required for multiclass classification.

## 4. LSTM architecture design

The first sequence model used in the project is a configurable Long Short-Term
Memory (LSTM) network. LSTM is a recurrent neural network architecture
designed to model dependencies across ordered timesteps.

The model processes the seven observations in each input sequence while
maintaining a hidden representation that is updated as the sequence is
processed. This allows information from neighboring and earlier timesteps to
contribute to the representation used for classification.

The basic architecture is:

    Input
      ↓
    LSTM
      ↓
    Dropout
      ↓
    TimeDistributed Dense + Softmax
      ↓
    Per-timestep class probabilities

### LSTM layer

The LSTM layer receives an input tensor with shape

    (batch, 7, 41)

and produces a sequence of hidden representations with shape

    (batch, 7, units)

where `units` is the number of hidden units in the LSTM layer.

The LSTM layer uses `return_sequences=True`. This is essential for the
seq2seq formulation because a representation must be retained for every
timestep. If `return_sequences=False` were used, the LSTM would return only
its final hidden representation, which would be incompatible with producing
one classification output for each of the seven timesteps.

### Dropout

A dropout layer is applied after each LSTM layer. During training, dropout
randomly sets a fraction of the hidden activations to zero. This acts as a
regularization mechanism and can reduce over-reliance on particular hidden
representations.

The dropout rate is configurable through the `dropout_rate` parameter.

### TimeDistributed classification head

The final classification head consists of a Dense layer with four output
units and a softmax activation, applied independently to each timestep.

Conceptually, the same classifier is applied to every LSTM output:

    h1 → Dense(4) → softmax → p1
    h2 → Dense(4) → softmax → p2
    ...
    h7 → Dense(4) → softmax → p7

The resulting output has shape

    (batch, 7, 4)

where the final dimension contains the estimated probabilities of the four
target classes.

The classifier therefore does not produce a single probability vector for
the complete sequence. It produces one probability vector for every
timestep.

### Loss function

The model is compiled using
`sparse_categorical_crossentropy`. This is consistent with the target
representation described in Section 3, where each timestep is represented by
an integer class ID rather than a one-hot vector.

The loss is evaluated across the sequence timesteps, allowing the model to
learn from the classification target associated with each timestep.

### Optimizer

The model uses the Adam optimizer. The learning rate is configurable through
the `learning_rate` parameter.

The initial implementation deliberately keeps the architecture relatively
simple. This provides a transparent baseline and avoids introducing
unnecessary architectural complexity before establishing model performance.
More complex recurrent configurations can subsequently be evaluated using
the same experiment and evaluation framework.

## 5. Configurable stacked LSTM architecture

The LSTM implementation is designed as a configurable model builder rather
than as separate implementations for single-layer and stacked architectures.
The number and size of recurrent layers are controlled through the
`lstm_units` parameter.

For example:

    lstm_units=(64,)

creates a single recurrent layer:

    Input
      ↓
    LSTM(64, return_sequences=True)
      ↓
    Dropout
      ↓
    TimeDistributed Dense(4) + Softmax

while:

    lstm_units=(64, 32)

creates two stacked recurrent layers:

    Input
      ↓
    LSTM(64, return_sequences=True)
      ↓
    Dropout
      ↓
    LSTM(32, return_sequences=True)
      ↓
    Dropout
      ↓
    TimeDistributed Dense(4) + Softmax

The same principle can be extended to additional layers, for example:

    lstm_units=(128, 64, 32)

would create three stacked LSTM layers.

### Why use a configurable architecture?

The configurable design allows model depth and hidden-layer size to be treated
as experimental hyperparameters without duplicating model-building code.
Consequently, a single-layer LSTM can serve as a simple baseline while
deeper configurations can be evaluated under the same training and evaluation
procedures.

This is preferable to maintaining separate model implementations for every
possible LSTM depth because the surrounding experiment pipeline remains
unchanged.

### Requirement for `return_sequences=True`

Every LSTM layer uses `return_sequences=True`, including the final recurrent
layer.

For stacked LSTMs, this is necessary because the output of one LSTM must
remain a sequence in order to be passed to the next LSTM layer. It is also
necessary for the final LSTM because the subsequent classification head must
receive a hidden representation for every timestep.

Thus, for a two-layer configuration:

    Input
      ↓
    LSTM(64, return_sequences=True)
      ↓
    Dropout
      ↓
    LSTM(32, return_sequences=True)
      ↓
    Dropout
      ↓
    TimeDistributed Dense(4) + Softmax

the intermediate and final tensor shapes are:

    Input:
        (batch, 7, 41)

    First LSTM:
        (batch, 7, 64)

    Second LSTM:
        (batch, 7, 32)

    Classification head:
        (batch, 7, 4)

### Model comparison principle

Increasing the number of LSTM layers increases the representational capacity
of the model, but also increases its number of trainable parameters and
potential for overfitting and unnecessary complexity.

Therefore, model depth is treated as an experimental choice rather than
assuming that a deeper architecture is automatically preferable. The same
validation and test procedures are used for different configurations so that
their performance can be compared under consistent conditions.

The initial configurations can therefore include both a simple baseline and
one or more stacked alternatives, while retaining exactly the same input
representation, target encoding, boundary handling, prediction aggregation,
and evaluation methodology.

## 6. Training strategy and early stopping

The model is trained using the training partition while the validation
partition is used for model selection and early stopping. The test partition
is not used during training or model selection.

This separation is maintained to prevent information from the validation or
test data from influencing the fitted model or the selection of its training
configuration.

### Training data

The model is fitted using:

    X_train
    y_train

The training targets contain integer class labels for every timestep in each
sequence.

Complete sequence windows may be shuffled between training batches using
`shuffle=True`. This does not change the temporal ordering of observations
within an individual sequence. For example, the window

    [t1, t2, t3, t4, t5, t6, t7]

remains in that order, although the complete window may be presented to the
model before or after another window.

This is compatible with the stateless LSTM architecture used here because
the temporal dependencies that the model learns are contained within each
sequence window.

### Validation data

After each training epoch, the model is evaluated on the validation
sequences. Validation predictions are not treated as independent timestep
observations because consecutive sequence windows overlap.

Instead, validation predictions undergo the same boundary trimming and
overlap aggregation procedure used for final evaluation. The resulting
unique timestep-level predictions are then used to calculate validation
macro-F1.

This produces the monitoring quantity:

    val_macro_f1

which represents macro-F1 on the retained original validation timesteps
after prediction aggregation.


## Training and Validation Macro-F1

### Definition of Macro-F1

For this project, `macro-F1` is defined at the **original-timestep level**, rather than directly over sequence positions.

Model predictions are processed as follows:

1. Generate predictions for each sequence.
2. Remove the boundary positions that are excluded from evaluation.
3. Map retained sequence positions back to their original timestamps using the sequence metadata.
4. Aggregate predictions from overlapping sequences referring to the same original timestep.
5. Compute the per-class F1 scores from the resulting timestep-level predictions.
6. Compute macro-F1 as the unweighted mean of the per-class F1 scores.

Therefore, macro-F1 represents performance across the original retained timesteps, with each class contributing equally to the final score.

### Training Macro-F1

`macro_f1` denotes the evaluation-consistent macro-F1 calculated on the training set after each epoch.

It uses the **same boundary trimming, overlap aggregation, and timestep-level evaluation procedure** as `val_macro_f1` and the final test macro-F1.

It is therefore not the ordinary Keras batch-level classification metric. Instead, after an epoch, the model's training-set predictions are processed through the same evaluation logic used for validation and test data.

The purpose of `macro_f1` is to provide a training-side counterpart to `val_macro_f1` that is directly comparable under the same definition.

The resulting learning-history fields are therefore:

- `macro_f1`: evaluation-consistent macro-F1 on the training set.
- `val_macro_f1`: evaluation-consistent macro-F1 on the validation set.

Comparing these two quantities helps identify potential overfitting while maintaining a consistent definition of macro-F1 across training, validation, and test evaluation.

### Computational Consideration

Because evaluation-consistent training macro-F1 requires generating predictions over the training set and performing boundary trimming and overlap aggregation after each epoch, it introduces additional computation compared with ordinary batch-level training metrics such as accuracy.

This additional computation is intentional: the resulting metric is directly comparable with the validation macro-F1 (in the learning curves) and test macro-F1 used by the project's evaluation protocol.



### Early stopping

Training uses Keras `EarlyStopping` with:

    monitor = "val_macro_f1"
    mode = "max"
    restore_best_weights = True

The objective is therefore to retain the model state that achieved the
highest validation macro-F1 rather than simply the model state from the last
training epoch.

The `patience` parameter specifies how many consecutive epochs without an
improvement in validation macro-F1 are tolerated before training is stopped.

For example, with:

    patience = 5

training can continue for up to five epochs without an improvement before
early stopping terminates the training process.

### Why validation macro-F1 is monitored

The target classes are not necessarily equally represented. Accuracy can
therefore be dominated by the more frequent classes and may provide an
incomplete description of performance on minority classes.

Macro-F1 addresses this by calculating F1 separately for each class and then
giving each class equal weight in the final average.

Consequently, validation macro-F1 is used as the primary model-selection
criterion, while accuracy remains a supplementary metric.

### Best-weight restoration

`restore_best_weights=True` ensures that, after early stopping, the model
contains the weights from the epoch with the highest validation macro-F1.

This is important because the final model should correspond to the selected
validation optimum rather than simply to the final epoch reached before
training stopped.

### Separation from the test set

The test set is not used to determine:

- when training stops,
- which epoch is selected,
- which model configuration is preferred,
- or any other training decision.

The test set is evaluated only after the model has been selected using the
training and validation data. This preserves the role of the test set as an
unseen dataset for the final performance assessment.

## 7. Validation macro-F1 computation

Validation macro-F1 is computed on the model's predictions after applying the
same boundary-trimming and overlap-aggregation procedure used for final
evaluation.

This is important because the model operates on overlapping sequence
windows, whereas the quantity of interest is the classification performance
at each unique original timestep.

### Why macro-F1 is computed after aggregation

With a stride of one, consecutive seven-timestep windows overlap strongly.
Consequently, the same original timestep can appear in several different
sequence windows and receive several predictions.

For example, with sequence length 7:

    Window 1: t1 t2 t3 t4 t5 t6 t7
    Window 2:    t2 t3 t4 t5 t6 t7 t8
    Window 3:       t3 t4 t5 t6 t7 t8 t9

The timestep t5 therefore occurs in all three windows.

If macro-F1 were calculated directly by flattening all sequence predictions,
t5 would contribute three times rather than once. This would make the metric
depend on the degree of window overlap rather than representing performance
on the original time series.

The validation procedure therefore first converts the overlapping sequence
predictions back into predictions for unique original timesteps.

### Validation prediction procedure

At the end of each training epoch, the following procedure is performed:

    Model
      ↓
    validation softmax probabilities
      ↓
    boundary trimming
      ↓
    group by (group, time)
      ↓
    average probability vectors
      ↓
    argmax
      ↓
    one prediction per retained original timestep
      ↓
    compare with true labels
      ↓
    macro-F1

The probability vectors are averaged before selecting the final class. This
preserves more information than averaging already-discretized class labels.

For a timestep receiving several probability vectors,

    p1, p2, ..., pk

the aggregated probability vector is

    p̄ = (1/k) Σ pi

and the final predicted class is the class with the largest component of p̄.

### Matching the true labels

The true labels are mapped to the same `(group, time)` identifiers used by
the aggregated predictions. When an original timestep occurs in multiple
overlapping windows, its true class should be identical in all occurrences.

If overlapping windows contain inconsistent true labels for the same
`(group, time)` pair, the evaluation procedure raises an error rather than
silently choosing one of the conflicting labels.

This provides an explicit consistency check on the sequence-generation and
label-alignment process.

### Macro-F1 calculation

After aggregation, the final arrays contain one true label and one predicted
label for each retained original timestep.

For each class, precision and recall are calculated and combined into an F1
score. Macro-F1 is then obtained by assigning equal weight to all classes:

    Macro-F1 = (1 / C) Σ F1_c

where C is the number of target classes.

Thus, each of the four risk states contributes equally to the final
macro-F1, regardless of its frequency in the evaluation data.

### Consistency with final evaluation

The same aggregation logic is used during validation and final test
evaluation. This ensures that the quantity used for early stopping,
`val_macro_f1`, is defined consistently with the macro-F1 reported for the
selected model on the test set.

The test set is not used while calculating validation macro-F1 or making
training decisions.

## 8. Overlapping sequence windows and prediction aggregation

The sequence dataset is constructed using a stride of one timestep. With a
sequence length of seven, consecutive windows therefore overlap strongly.

For example, a sequence of ten timesteps produces windows such as:

    Window 1: t1 t2 t3 t4 t5 t6 t7
    Window 2:    t2 t3 t4 t5 t6 t7 t8
    Window 3:       t3 t4 t5 t6 t7 t8 t9
    Window 4:          t4 t5 t6 t7 t8 t9 t10

Consequently, an interior timestep can occur in several different sequence
windows and can receive several model predictions.

### Why overlapping windows are useful

Overlapping windows allow each timestep to be considered in multiple temporal
contexts. This is particularly useful for boundary-sensitive sequence
classification because predictions made at the edge of a window have less
context available within that window than predictions near its center.

For example, t5 in the following window has substantial context on both
sides:

    t2 t3 t4 [t5] t6 t7 t8

whereas t2 is at the beginning of the same window:

    [t2] t3 t4 t5 t6 t7 t8

The overlapping-window strategy therefore provides multiple contextual
views of many original timesteps.

### Prediction aggregation

The model produces a probability vector for every timestep of every sequence.
These predictions are not treated as independent final predictions.

After boundary trimming, predictions referring to the same original
`(group, time)` pair are collected together. Their class-probability vectors
are averaged:

    p̄(group, time) = (1 / k) Σ p_i(group, time)

where k is the number of retained sequence predictions associated with that
original timestep.

The final predicted class is then obtained by applying `argmax` to the
averaged probability vector:

    ŷ(group, time) = argmax p̄(group, time)

This produces exactly one final prediction for each retained original
timestep.

### Why probabilities are averaged before classification

The aggregation is performed on the softmax probability vectors rather than
on already-discretized class labels.

For example, suppose two overlapping windows produce:

    [0.55, 0.45]
    [0.40, 0.60]

for the same timestep.

Their average is:

    [0.475, 0.525]

and the resulting prediction is class 1.

Averaging class labels instead would discard the confidence information
contained in the probability vectors and could produce less informative
aggregation behavior.

### Metadata required for aggregation

Each generated sequence therefore carries metadata identifying its original
temporal location. In particular, the metadata records:

- the sequence's group identifier;
- the ordered timestamps corresponding to the sequence positions.

This allows predictions from overlapping windows to be mapped back to their
original `(group, time)` locations after model inference.

### One prediction per original timestep

The purpose of aggregation is to transform the model's window-level output
representation:

    (sequence, position, class)

into a timestep-level representation:

    (group, time, class probabilities)

This distinction is important for downstream evaluation and reporting.
Metrics should represent performance on the original retained timesteps,
rather than giving additional weight to timesteps merely because they happen
to occur in more overlapping windows.

## 9. Boundary trimming strategy

Although overlapping windows provide multiple temporal contexts for many
timesteps, predictions near sequence boundaries are less reliable because
those positions have less neighboring context within the individual window.

For this reason, a `boundary_width` parameter is used during prediction
aggregation. With the current configuration:

    sequence_length = 7
    boundary_width = 2

the first two and last two positions of every sequence window are excluded
from the final prediction set.

The retained positions are therefore:

    Position:    1   2   3   4   5   6   7
                 X   X   ✓   ✓   ✓   X   X

Only the central three positions contribute predictions to the aggregated
timestep-level output.

### Why boundary predictions are discarded

Consider the following window:

    t1 t2 t3 t4 t5 t6 t7

The central timestep t4 has neighboring observations on both sides within the
window:

    t1 t2 t3 [t4] t5 t6 t7

In contrast, t1 is located at the beginning of the window and has no earlier
observation available within that window:

    [t1] t2 t3 t4 t5 t6 t7

Similarly, t7 has no later observation within the window.

Because the model's hidden representation depends on the temporal context
available to it, predictions near the boundaries are treated as less
well-supported than predictions near the center of a window.

### Why the first and last original timesteps are also excluded

A special case could be introduced for the first sequence window and the last
sequence window so that their outer boundary positions are retained.
However, these positions still have less temporal context than interior
timesteps.

The first timesteps of an account also have a related edge effect in
feature engineering: lag-based features cannot have a genuine previous
observation before the beginning of the account history. Thus, the earliest
observations may already have less historical information available.

For consistency, the project therefore uses the same boundary-trimming rule
for every sequence rather than introducing special treatment for the first
and last sequence windows.

### Consequence for prediction coverage

Boundary trimming means that the final prediction table does not contain a
prediction for every original timestep.

For example, with a seven-timestep sequence and `boundary_width = 2`, only
the three central positions of each sequence contribute predictions.
Consequently, the first and last two timesteps of each account history are
not expected to appear in the final aggregated prediction set.

These missing predictions are intentional and must not be interpreted as
model failures or missing inference results.

### Consequences for downstream processing

Downstream evaluation and reporting must operate on the retained aggregated
prediction set rather than assuming that every original `(group, time)` pair
has a prediction.

In particular:

- evaluation metrics are calculated only on retained timesteps;
- confusion matrices use only retained predictions;
- prediction tables may contain fewer timesteps than the original data;
- joins involving predictions must account for intentionally absent boundary
  timesteps;
- missing predictions at these boundaries must not be converted into an
  artificial prediction or treated as an error.

The same boundary width must be used consistently during validation and final
test evaluation so that `val_macro_f1` and test macro-F1 are calculated under
the same definition.

The boundary-trimming rule is therefore part of the model's evaluation
contract rather than merely an implementation detail of the prediction
aggregation function.

## 10. Final evaluation procedure

The final evaluation is performed on the test partition only after model
training, early stopping, and model selection have been completed using the
training and validation partitions.

The test data therefore remains unseen during all model-selection decisions.

### Evaluation workflow

The final evaluation follows the same prediction-processing procedure used
for validation:

    Test sequences
        ↓
    trained model
        ↓
    softmax probabilities
        ↓
    boundary trimming
        ↓
    overlap aggregation
        ↓
    one prediction per retained original timestep
        ↓
    comparison with true labels
        ↓
    final evaluation metrics

This ensures that the metric used for final evaluation is defined in the same
way as the validation metric used during training.

### Test predictions

The trained model produces a probability vector for each timestep of each
test sequence. These probabilities are passed to the prediction-aggregation
procedure.

Predictions at sequence boundaries are removed according to the configured
`boundary_width`. Predictions referring to the same original `(group, time)`
pair are then combined by averaging their probability vectors.

The final class for each retained timestep is obtained from the class with
the highest aggregated probability.

### Metrics

The final evaluation reports at least:

- **Accuracy**
- **Macro-F1**
- **Confusion matrix**

Accuracy measures the proportion of retained timesteps whose predicted class
matches the true class.

Macro-F1 is calculated independently for each target class and then averaged
with equal weight across classes. This prevents the more frequent classes
from dominating the summary metric.

The confusion matrix provides a class-by-class view of the errors. Its rows
represent the true classes and its columns represent the predicted classes.

For four classes, the matrix has the form:

                    Predicted
                 0    1    2    3
    True  0     [ ]  [ ]  [ ]  [ ]
          1     [ ]  [ ]  [ ]  [ ]
          2     [ ]  [ ]  [ ]  [ ]
          3     [ ]  [ ]  [ ]  [ ]

### Evaluation population

The reported metrics apply only to the original timesteps that remain after
boundary trimming and overlap aggregation.

They therefore should not be interpreted as metrics over every original
timestep in the raw dataset.

The number of evaluated timesteps is recorded explicitly as
`n_predictions` in the evaluation result so that the population on which the
metrics were calculated is transparent.

### Validation versus test evaluation

Validation evaluation is used during training to determine when the model
should stop and which model weights should be retained.

Test evaluation is performed only after the model has been selected.

The test metrics therefore provide an estimate of performance on an
unseen partition under the same timestep-level evaluation procedure used
during validation.

No test metric is used to modify the model, select the number of epochs,
select the best checkpoint, or tune hyperparameters.

### Reproducibility of evaluation

The same:

- sequence length,
- boundary width,
- class definition,
- prediction aggregation rule,
- true-label alignment procedure,
- and metric definitions

must be used for validation and test evaluation.

Changing these evaluation rules between validation and test data would make
the resulting metrics non-comparable and could invalidate the interpretation
of model-selection results.


## 11. Experiment orchestration workflow

The complete model experiment is orchestrated by `experiment.py`. Its purpose
is to provide a single, reproducible workflow that connects model
construction, training, prediction, aggregation, and evaluation without
duplicating these responsibilities across separate scripts.

The experiment runner performs the following sequence:

    Prepared model dataset
            ↓
    Build configurable LSTM
            ↓
    Compile model
            ↓
    Train on training partition
            ↓
    Compute aggregated validation macro-F1
            ↓
    Early stopping and best-weight restoration
            ↓
    Generate validation predictions
            ↓
    Trim boundaries and aggregate overlaps
            ↓
    Evaluate validation performance
            ↓
    Generate test predictions
            ↓
    Trim boundaries and aggregate overlaps
            ↓
    Evaluate test performance
            ↓
    Return model, training history, and evaluation results

### Separation of responsibilities

The modeling implementation is divided into several modules, each with a
specific responsibility.

`lstm.py` defines the configurable LSTM architecture. It is responsible for
constructing and compiling the neural network but does not determine how an
experiment is evaluated.

`training.py` defines the training procedure. It handles model fitting,
validation macro-F1 calculation during training, early stopping, and
restoration of the best model weights.

`metrics.py` contains the implementation of the multiclass macro-F1 metric.
Keeping the metric separate makes its definition reusable and independently
testable.

`prediction_aggregation.py` converts overlapping sequence predictions into
one prediction per retained original timestep. It handles boundary trimming,
grouping by original temporal location, probability averaging, and final
class selection.

`evaluation.py` applies the aggregation procedure and calculates the final
evaluation metrics, including accuracy, macro-F1, and the confusion matrix.

`experiment.py` connects these components into one complete experiment
workflow.

This separation avoids embedding training, prediction aggregation, and
evaluation logic directly inside the model architecture.

### Validation and test separation

The experiment runner uses the validation partition during training for
early stopping and model selection. After training has finished and the best
weights have been restored, the same trained model is evaluated on both the
validation and test partitions.

The test partition does not influence the training process or model
selection.

```Pyhton
dataset
├── X_train
├── y_train
├── train_metadata
├── X_validation
├── y_validation
├── validation_metadata
├── X_test
├── y_test
├── test_metadata
├── final_feature_columns
├── train_end
└── validation_end
```


### Returned experiment result

The experiment runner returns a structured result containing:

- the trained Keras model;
- the Keras training history;
- the aggregated validation evaluation;
- the aggregated test evaluation.

The evaluation results contain the final predictions, true labels, metrics,
confusion matrix, and number of retained predictions.

This structure allows subsequent scripts to save the trained model, export
metrics, generate visualizations, or perform further analysis without
reimplementing the experiment itself.

### Reusability

The experiment runner is designed around the model-independent training and
evaluation interfaces wherever possible. Consequently, future sequence
models can reuse the same prediction aggregation and evaluation procedures
provided that they produce per-timestep class probabilities with the same
general output structure.

This separation also makes controlled model comparisons easier: different
LSTM configurations can be supplied to the same experiment workflow without
changing the downstream evaluation procedure.

## 12. Reproducibility and future model extensions

The modeling workflow is designed so that model experiments can be repeated
under explicitly defined configurations while keeping the data preparation,
training, prediction aggregation, and evaluation procedures consistent.

### Reproducible experiment configuration

An experiment is defined by its model and training parameters, including:

- sequence length;
- number of input features;
- number of target classes;
- LSTM layer configuration;
- dropout rate;
- learning rate;
- batch size;
- maximum number of epochs;
- early-stopping patience;
- prediction boundary width.

These parameters should be recorded for each experiment so that the resulting
model and performance metrics can be traced to a specific configuration.

Random seeds should also be controlled where reproducibility is required.
However, exact numerical reproducibility can depend on the TensorFlow,
hardware, operating-system, and execution environment.

### Consistent evaluation across model configurations

When different model configurations are compared, the following should remain
unchanged unless the experiment explicitly investigates one of them:

- model-dataset construction;
- training, validation, and test partitions;
- feature encoding;
- training-only preprocessing statistics;
- sequence length and stride;
- target encoding;
- boundary-trimming rule;
- overlap-aggregation procedure;
- evaluation population;
- metric definitions.

Changing several of these factors simultaneously would make it difficult to
attribute differences in performance to the model architecture itself.

### Future model extensions

The current experiment framework is centered on an LSTM sequence classifier,
but the surrounding components are intended to be reusable.

Potential future sequence models include:

- deeper or differently sized LSTM configurations;
- GRU-based sequence models;
- temporal convolutional networks (TCNs);
- Transformer-based sequence models.

A future model implementation should ideally preserve the same general output
contract:

    (batch, sequence_length, n_classes)

where the final dimension contains per-timestep class probabilities.

Models following this contract can reuse the existing prediction aggregation
and evaluation procedures without changing the definition of the final
timestep-level metrics.

### Controlled model experimentation

Model extensions should be introduced as explicit experimental
configurations rather than modifying the baseline architecture without
recording the change.

For example, different LSTM configurations can be represented through:

    lstm_units=(64,)

    lstm_units=(64, 32)

    lstm_units=(128, 64)

and evaluated using the same training and evaluation workflow.

This makes architectural comparisons easier to reproduce and keeps the
experimental process separate from the implementation of individual model
components.

### Test coverage

The model-related components are independently tested before being used for
full-scale experiments. The tests cover, among other aspects:

- model construction and output shapes;
- single-layer and stacked LSTM configurations;
- sparse multiclass target handling;
- macro-F1 calculation;
- boundary trimming;
- overlapping prediction aggregation;
- true-label alignment;
- confusion-matrix construction;
- validation macro-F1 computation;
- early-stopping training;
- complete experiment orchestration.

This test structure is intended to detect changes in the data-to-prediction
pipeline before they affect the results of computationally more expensive
model experiments.