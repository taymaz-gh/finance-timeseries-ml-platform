from finance_ml.data.generate_synthetic import (
    generate_financial_timeseries,
)


def test_generated_row_count() -> None:
    """Checking that each account receives one row per simulated day."""
    df = generate_financial_timeseries(
        n_accounts=4,
        n_days=10,
    )

    assert len(df) == 40


def test_generated_risk_states() -> None:
    """Checking that generated risk-state labels are valid."""
    df = generate_financial_timeseries(
        n_accounts=10,
        n_days=20,
    )

    assert df["risk_state"].isin([0, 1, 2, 3]).all()


def test_account_dates_are_unique() -> None:
    """Checking that clean source data has one row per account and date."""
    df = generate_financial_timeseries(
        n_accounts=5,
        n_days=10,
    )

    duplicates = df.duplicated(subset=["account_id", "date"])

    assert not duplicates.any()


def test_generation_is_reproducible() -> None:
    """Checking that the same random seed reproduces identical data."""
    df_1 = generate_financial_timeseries(
        n_accounts=5,
        n_days=10,
        random_seed=42,
    )

    df_2 = generate_financial_timeseries(
        n_accounts=5,
        n_days=10,
        random_seed=42,
    )

    assert df_1.equals(df_2)


def test_default_generation_contains_all_risk_states() -> None:
    """Checking that the default simulation generates all four classes."""
    df = generate_financial_timeseries()

    observed_states = set(df["risk_state"].unique())

    assert observed_states == {0, 1, 2, 3}


from finance_ml.data.inject_quality_issues import (
    inject_data_quality_issues,
)


def test_quality_issues_are_injected() -> None:
    """Checking that Bronze-style corruption introduces expected issues."""
    clean_df = generate_financial_timeseries(
        n_accounts=100,
        n_days=20,
    )

    bronze_df = inject_data_quality_issues(
        clean_df,
    )

    assert len(bronze_df) > len(clean_df)

    assert bronze_df.isna().any().any()

    assert (bronze_df["cash_outflow"] < 0).any()

    assert (bronze_df["credit_utilization"] > 1).any()

    assert bronze_df.duplicated(subset=["account_id", "date"]).any()


def test_quality_issue_injection_is_reproducible() -> None:
    """Checking that identical seeds produce identical Bronze data."""
    clean_df = generate_financial_timeseries(
        n_accounts=20,
        n_days=10,
    )

    bronze_1 = inject_data_quality_issues(
        clean_df,
        random_seed=123,
    )

    bronze_2 = inject_data_quality_issues(
        clean_df,
        random_seed=123,
    )

    assert bronze_1.equals(bronze_2)
