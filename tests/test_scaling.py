import pytest
from pyspark.sql import SparkSession

from finance_ml.preprocessing.scaling import (
    apply_scaling,
    fit_scaler,
)




def test_fit_standard_scaler_uses_training_statistics(
    spark,
) -> None:
    """Checking mean/std scaling fitted from training data."""
    train_df = spark.createDataFrame(
        [
            (10.0,),
            (20.0,),
            (30.0,),
        ],
        ["value"],
    )

    parameters = fit_scaler(
        train_df,
        ["value"],
        method="standard",
    )

    assert parameters["value"]["center"] == pytest.approx(20.0)
    assert parameters["value"]["scale"] == pytest.approx((200.0 / 3.0) ** 0.5)
    assert parameters["value"]["method"] == "standard"


def test_fit_mad_scaler_uses_median_and_corrected_mad(
    spark,
) -> None:
    """Checking robust scaling based on median and corrected MAD."""
    train_df = spark.createDataFrame(
        [
            (1.0,),
            (2.0,),
            (3.0,),
            (100.0,),
        ],
        ["value"],
    )

    parameters = fit_scaler(
        train_df,
        ["value"],
        method="mad",
    )

    # Spark percentile_approx returns a discrete approximate median here.
    assert parameters["value"]["center"] == pytest.approx(2.0)

    # Absolute deviations from 2 are:
    # [1, 0, 1, 98]
    # MAD = 1
    # corrected MAD = 1.4826
    assert parameters["value"]["scale"] == pytest.approx(1.4826)

    assert parameters["value"]["method"] == "mad"


def test_apply_scaling_uses_fitted_parameters(
    spark,
) -> None:
    """Checking application of already-fitted scaling parameters."""
    df = spark.createDataFrame(
        [
            (10.0,),
            (20.0,),
            (30.0,),
        ],
        ["value"],
    )

    parameters = {
        "value": {
            "center": 20.0,
            "scale": 10.0,
            "method": "standard",
        }
    }

    transformed_df = apply_scaling(
        df,
        parameters,
    )

    values = [row["value"] for row in transformed_df.collect()]

    assert values == pytest.approx([-1.0, 0.0, 1.0])


def test_standard_scaled_training_feature_has_zero_mean_and_unit_std(
    spark,
) -> None:
    """
    Checking that standard scaling gives approximately zero mean
    and unit population standard deviation on training data.
    """
    from pyspark.sql import functions as F

    train_df = spark.createDataFrame(
        [
            (10.0,),
            (20.0,),
            (30.0,),
        ],
        ["value"],
    )

    parameters = fit_scaler(
        train_df,
        ["value"],
        method="standard",
    )

    transformed_df = apply_scaling(
        train_df,
        parameters,
    )

    statistics = transformed_df.select(
        F.avg("value").alias("mean"),
        F.stddev_pop("value").alias("std"),
    ).first()

    assert statistics["mean"] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert statistics["std"] == pytest.approx(
        1.0,
    )


def test_mad_scaling_is_robust_to_outlier(
    spark,
) -> None:
    """
    Checking that an extreme value does not strongly shift the
    MAD-based center.
    """
    train_df = spark.createDataFrame(
        [
            (1.0,),
            (2.0,),
            (3.0,),
            (4.0,),
            (1000.0,),
        ],
        ["value"],
    )

    parameters = fit_scaler(
        train_df,
        ["value"],
        method="mad",
    )

    assert parameters["value"]["center"] == pytest.approx(3.0)
    assert parameters["value"]["scale"] == pytest.approx(1.4826)


def test_zero_variance_standard_feature_scales_to_zero(
    spark,
) -> None:
    """Checking safe handling of a constant feature."""
    train_df = spark.createDataFrame(
        [
            (10.0,),
            (10.0,),
            (10.0,),
        ],
        ["value"],
    )

    parameters = fit_scaler(
        train_df,
        ["value"],
        method="standard",
    )

    assert parameters["value"]["center"] == 10.0
    assert parameters["value"]["scale"] == 1.0

    transformed_df = apply_scaling(
        train_df,
        parameters,
    )

    values = [row["value"] for row in transformed_df.collect()]

    assert values == [0.0, 0.0, 0.0]


def test_zero_mad_falls_back_to_standard_deviation(
    spark,
) -> None:
    """
    Checking that MAD=0 falls back to non-zero training standard deviation.
    """
    train_df = spark.createDataFrame(
        [
            (0.0,),
            (0.0,),
            (0.0,),
            (10.0,),
        ],
        ["value"],
    )

    parameters = fit_scaler(
        train_df,
        ["value"],
        method="mad",
    )

    assert parameters["value"]["center"] == 0.0

    # Population std of [0, 0, 0, 10].
    expected_std = (18.75) ** 0.5

    assert parameters["value"]["scale"] == pytest.approx(expected_std)


def test_invalid_scaling_method_raises_error(
    spark,
) -> None:
    """Checking validation of unsupported scaling strategies."""
    train_df = spark.createDataFrame(
        [
            (1.0,),
            (2.0,),
            (3.0,),
        ],
        ["value"],
    )

    with pytest.raises(
        ValueError,
        match="method must be either 'standard' or 'mad'",
    ):
        fit_scaler(
            train_df,
            ["value"],
            method="unknown",
        )
