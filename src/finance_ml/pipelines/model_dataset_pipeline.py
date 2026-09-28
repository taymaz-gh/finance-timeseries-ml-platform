"""End-to-end preparation of model-ready time-series datasets."""

from __future__ import annotations

from finance_ml.preprocessing.encoding import (
    apply_one_hot_encoding,
    fit_category_levels,
)
from finance_ml.preprocessing.feature_schema import (
    build_final_feature_columns,
)
from finance_ml.preprocessing.imputation import (
    apply_numeric_imputation,
    fit_numeric_medians,
)
from finance_ml.preprocessing.model_dataset import (
    get_model_columns,
)
from finance_ml.preprocessing.scaling import (
    apply_scaling,
    fit_scaler,
)
from finance_ml.preprocessing.sequences import (
    build_sequences,
)
from finance_ml.preprocessing.split import (
    temporal_split,
)


def prepare_model_dataset(
    gold_df,
    *,
    time_col: str = "date",
    group_col: str = "account_id",
    target_col: str = "risk_state",
    target_name_col: str = "risk_state_name",
    categorical_columns: tuple[str, ...] = ("account_type",),
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    sequence_length: int = 7,
    stride: int = 1,
    scaling_method: str = "mad",
):
    """
    Prepare leakage-safe train, validation, and test sequence datasets.

    The pipeline performs:

    1. chronological splitting
    2. feature-schema identification
    3. train-fitted numeric median imputation
    4. missing-indicator creation
    5. train-fitted categorical level discovery
    6. one-hot encoding
    7. final numerical feature-schema construction
    8. train-fitted feature scaling
    9. model-column selection
    10. sequence construction

    Args:
        gold_df:
            Gold-layer Spark DataFrame.
        time_col:
            Timestamp/date column used for temporal ordering and splitting.
        group_col:
            Entity identifier used to separate independent time series.
        target_col:
            Integer target-label column.
        target_name_col:
            Human-readable target-label column.
        categorical_columns:
            Categorical predictor columns.
        train_fraction:
            Fraction of distinct timestamps assigned to training.
        validation_fraction:
            Fraction of distinct timestamps assigned to validation.
        sequence_length:
            Number of timesteps in each sequence.
        stride:
            Number of rows to advance between consecutive sequence windows.
        scaling_method:
            Scaling strategy for continuous numeric features.

            Supported values:

            - "mad":
              median-centered robust scaling using
              1.4826 * median absolute deviation.

            - "standard":
              mean-centered scaling using the population
              standard deviation.

            Scaling parameters are always fitted on the training split only.

    Returns:
        Dictionary containing:
            - train, validation, and test sequence tensors
            - sequence metadata
            - model-ready Spark DataFrames
            - final feature columns
            - temporal split boundaries
            - fitted imputation medians
            - fitted categorical levels
            - fitted scaling parameters
            - model-column metadata
    """
    (
        train_df,
        validation_df,
        test_df,
        train_end,
        validation_end,
    ) = temporal_split(
        gold_df,
        time_col=time_col,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    model_columns = get_model_columns(
        gold_df,
        target_col=target_col,
        target_name_col=target_name_col,
        id_columns=(
            group_col,
            time_col,
        ),
        categorical_columns=categorical_columns,
    )

    numeric_features = model_columns["numeric_feature_columns"]

    categorical_features = model_columns["categorical_feature_columns"]

    # Fitting numeric imputation statistics using training data only.
    medians = fit_numeric_medians(
        train_df,
        numeric_features,
    )

    train_imputed_df = apply_numeric_imputation(
        train_df,
        medians,
    )

    validation_imputed_df = apply_numeric_imputation(
        validation_df,
        medians,
    )

    test_imputed_df = apply_numeric_imputation(
        test_df,
        medians,
    )

    # Fitting categorical levels using training data only.
    category_levels = fit_category_levels(
        train_imputed_df,
        categorical_features,
    )

    train_encoded_df = apply_one_hot_encoding(
        train_imputed_df,
        category_levels,
    )

    validation_encoded_df = apply_one_hot_encoding(
        validation_imputed_df,
        category_levels,
    )

    test_encoded_df = apply_one_hot_encoding(
        test_imputed_df,
        category_levels,
    )

    # Building the final numerical model-input schema.
    final_feature_columns = build_final_feature_columns(
        numeric_features,
        medians,
        category_levels,
    )

    # Fitting scaling parameters using training data only.
    scaler_parameters = fit_scaler(
        train_encoded_df,
        numeric_features,
        method=scaling_method,
    )

    train_scaled_df = apply_scaling(
        train_encoded_df,
        scaler_parameters,
    )

    validation_scaled_df = apply_scaling(
        validation_encoded_df,
        scaler_parameters,
    )

    test_scaled_df = apply_scaling(
        test_encoded_df,
        scaler_parameters,
    )

    selected_columns = [
        group_col,
        time_col,
        *final_feature_columns,
        target_col,
        target_name_col,
    ]

    model_train_df = train_scaled_df.select(*selected_columns)

    model_validation_df = validation_scaled_df.select(*selected_columns)

    model_test_df = test_scaled_df.select(*selected_columns)

    (
        X_train,
        y_train,
        train_metadata,
    ) = build_sequences(
        model_train_df,
        feature_columns=final_feature_columns,
        target_col=target_col,
        group_col=group_col,
        time_col=time_col,
        sequence_length=sequence_length,
        stride=stride,
    )

    (
        X_validation,
        y_validation,
        validation_metadata,
    ) = build_sequences(
        model_validation_df,
        feature_columns=final_feature_columns,
        target_col=target_col,
        group_col=group_col,
        time_col=time_col,
        sequence_length=sequence_length,
        stride=stride,
    )

    (
        X_test,
        y_test,
        test_metadata,
    ) = build_sequences(
        model_test_df,
        feature_columns=final_feature_columns,
        target_col=target_col,
        group_col=group_col,
        time_col=time_col,
        sequence_length=sequence_length,
        stride=stride,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test,
        "train_metadata": train_metadata,
        "validation_metadata": validation_metadata,
        "test_metadata": test_metadata,
        "model_train_df": model_train_df,
        "model_validation_df": model_validation_df,
        "model_test_df": model_test_df,
        "final_feature_columns": final_feature_columns,
        "model_columns": model_columns,
        "train_end": train_end,
        "validation_end": validation_end,
        "medians": medians,
        "category_levels": category_levels,
        "scaler_parameters": scaler_parameters,
        "scaling_method": scaling_method,
    }
