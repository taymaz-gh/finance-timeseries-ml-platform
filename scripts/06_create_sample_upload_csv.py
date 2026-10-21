"""Generating sample raw financial observations for API testing."""

from pathlib import Path

from finance_ml.data.generate_synthetic import generate_financial_timeseries


def main() -> None:
    """Generating and exporting an unlabeled financial CSV."""

    # Generating reproducible synthetic account-day observations.
    df = generate_financial_timeseries(
        n_accounts=3,
        n_days=14,
        start_date="2026-01-01",
        random_seed=123,
    )

    # Removing target labels to prevent inference-time leakage.
    df = df.drop(columns=["risk_state", "risk_state_name"])

    # Creating the output directory.
    output_path = Path("artifacts/sample_accounts.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Saving raw financial observations.
    df.to_csv(output_path, index=False)

    print(f"CSV saved to: {output_path.resolve()}")
    print(f"Accounts: {df['account_id'].nunique()}")
    print(f"Observations: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
