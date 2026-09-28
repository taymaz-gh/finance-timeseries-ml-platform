from finance_ml.preprocessing.feature_schema import (
    build_final_feature_columns,
)


def test_build_final_feature_columns() -> None:
    """Checking construction of the final numerical feature schema."""

    numeric_features = [
        "balance",
        "cash_outflow",
        "payment_ratio",
    ]

    medians = {
        "cash_outflow": 100.0,
        "payment_ratio": 0.8,
    }

    category_levels = {
        "account_type": [
            "business",
            "checking",
            "credit",
        ]
    }

    result = build_final_feature_columns(
        numeric_features,
        medians,
        category_levels,
    )

    assert result == [
        "balance",
        "cash_outflow",
        "payment_ratio",
        "cash_outflow_missing",
        "payment_ratio_missing",
        "account_type_business",
        "account_type_checking",
        "account_type_credit",
    ]
