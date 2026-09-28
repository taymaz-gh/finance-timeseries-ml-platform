"""Leakage-safe feature scaling utilities for model datasets."""

from __future__ import annotations


def fit_scaler(
    train_df,
    columns: list[str],
    method: str = "mad",
) -> dict[str, dict[str, float]]:
    """
    Fit scaling parameters using training data only.

    Supported methods:

    - "standard":
        center = mean
        scale = population standard deviation

    - "mad":
        center = median
        scale = 1.4826 * MAD

        If corrected MAD is zero, the scaler falls back to the
        training population standard deviation. If that is also zero,
        the scale is set to 1.0.

    Args:
        train_df:
            Training Spark DataFrame.
        columns:
            Continuous numeric columns to scale.
        method:
            Scaling strategy: "standard" or "mad".

    Returns:
        Dictionary mapping each column to:
            - center
            - scale
            - method

    Raises:
        ValueError:
            If the method is unsupported, a requested column is missing,
            or valid scaling statistics cannot be computed.
    """
    from pyspark.sql import functions as F

    if method not in {"standard", "mad"}:
        raise ValueError("method must be either 'standard' or 'mad'.")

    for column in columns:
        if column not in train_df.columns:
            raise ValueError(
                f"Scaling column '{column}' does not exist in the DataFrame."
            )

    scaler_parameters: dict[
        str,
        dict[str, float | str],
    ] = {}

    if method == "standard":
        aggregations = []

        for column in columns:
            aggregations.extend(
                [
                    F.avg(column).alias(f"{column}__mean"),
                    F.stddev_pop(column).alias(f"{column}__std"),
                ]
            )

        statistics_row = train_df.select(*aggregations).first()

        for column in columns:
            center = statistics_row[f"{column}__mean"]

            scale = statistics_row[f"{column}__std"]

            if center is None or scale is None:
                raise ValueError(f"Cannot fit scaler for '{column}'.")

            if scale == 0:
                scale = 1.0

            scaler_parameters[column] = {
                "center": float(center),
                "scale": float(scale),
                "method": method,
            }

    else:
        for column in columns:
            median = train_df.select(
                F.percentile_approx(
                    column,
                    0.5,
                ).alias("median")
            ).first()["median"]

            if median is None:
                raise ValueError(f"Cannot compute median for '{column}'.")

            mad = train_df.select(
                F.percentile_approx(
                    F.abs(F.col(column) - F.lit(median)),
                    0.5,
                ).alias("mad")
            ).first()["mad"]

            if mad is None:
                raise ValueError(f"Cannot compute MAD for '{column}'.")

            scale = 1.4826 * float(mad)

            if scale == 0:
                std_value = train_df.select(F.stddev_pop(column).alias("std")).first()[
                    "std"
                ]

                if std_value is None:
                    raise ValueError(f"Cannot compute fallback scale for '{column}'.")

                scale = float(std_value) if std_value != 0 else 1.0

            scaler_parameters[column] = {
                "center": float(median),
                "scale": float(scale),
                "method": method,
            }

    return scaler_parameters


def apply_scaling(
    df,
    scaler_parameters,
):
    """
    Apply training-derived scaling parameters.

    Args:
        df:
            Spark DataFrame to transform.
        scaler_parameters:
            Parameters produced by `fit_scaler()`.

    Returns:
        Spark DataFrame with requested columns scaled.
    """
    from pyspark.sql import functions as F

    result_df = df

    for column, parameters in scaler_parameters.items():
        if column not in result_df.columns:
            raise ValueError(
                f"Scaling column '{column}' does not exist in the DataFrame."
            )

        center = parameters["center"]
        scale = parameters["scale"]

        result_df = result_df.withColumn(
            column,
            (F.col(column) - F.lit(center)) / F.lit(scale),
        )

    return result_df
