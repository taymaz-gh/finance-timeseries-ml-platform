"""Generating synthetic financial account time-series data."""

from __future__ import annotations

import numpy as np
import pandas as pd


RISK_STATE_NAMES = {
    0: "normal",
    1: "elevated_risk",
    2: "liquidity_stress",
    3: "delinquency_risk",
}


def _stress_to_state(stress: float) -> int:
    """Converting a latent financial-stress score into a risk-state class."""
    if stress < 0.35:
        return 0
    if stress < 0.55:
        return 1
    if stress < 0.75:
        return 2
    return 3


def generate_financial_timeseries(
    n_accounts: int = 2000,
    n_days: int = 60,
    start_date: str = "2026-01-01",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generating synthetic daily financial-account observations.

    Args:
        n_accounts:
            Number of independent customer accounts to simulate.
        n_days:
            Number of consecutive daily observations per account.
        start_date:
            First date in each account time series.
        random_seed:
            Seed used to make the simulation reproducible.

    Returns:
        A pandas DataFrame containing one row per account and day.
    """
    rng = np.random.default_rng(random_seed)

    dates = pd.date_range(
        start=start_date,
        periods=n_days,
        freq="D",
    )

    records: list[dict[str, object]] = []

    for account_number in range(1, n_accounts + 1):
        account_id = f"ACC_{account_number:05d}"

        account_type = rng.choice(
            ["checking", "credit", "business"],
            p=[0.50, 0.30, 0.20],
        )

        
        baseline_stress = float(
            rng.choice(
                [0.15, 0.40, 0.65, 0.85],
                p=[0.55, 0.25, 0.15, 0.05],
            )
        )

        stress = float(
            np.clip(
                rng.normal(
                    loc=baseline_stress,
                    scale=0.08,
                ),
                0.0,
                1.0,
            )
        )

        balance = float(
            rng.lognormal(
                mean=np.log(5000),
                sigma=0.45,
            )
        )

        credit_limit = float(rng.uniform(3000, 15000))

        for date in dates:
            # Evolving the latent financial-stress process.
            shock = rng.normal(
                loc=0.0,
                scale=0.05,
            )

            if rng.random() < 0.015:
                shock += rng.uniform(0.15, 0.35)
            
            stress = float(
                np.clip(
                    0.85 * stress
                    + 0.15 * baseline_stress
                    + shock,
                    0.0,
                    1.0,
                )
            )

            risk_state = _stress_to_state(stress)

            # Generating observed financial behavior conditional on stress.
            cash_inflow = max(
                0.0,
                rng.normal(
                    loc=180.0 * (1.0 - 0.45 * stress),
                    scale=60.0,
                ),
            )

            cash_outflow = max(
                0.0,
                rng.normal(
                    loc=150.0 * (1.0 + 0.55 * stress),
                    scale=55.0,
                ),
            )

            balance = max(
                0.0,
                balance + cash_inflow - cash_outflow,
            )

            transaction_count = max(
                0,
                int(
                    rng.poisson(
                        lam=max(
                            2.0,
                            10.0 - 3.0 * stress,
                        )
                    )
                ),
            )

            credit_utilization = float(
                np.clip(
                    rng.normal(
                        loc=0.25 + 0.65 * stress,
                        scale=0.08,
                    ),
                    0.0,
                    1.0,
                )
            )

            amount_due = max(
                20.0,
                credit_limit * credit_utilization * rng.uniform(0.015, 0.040),
            )

            expected_payment_ratio = np.clip(
                1.05 - 0.90 * stress,
                0.05,
                1.0,
            )

            payment_ratio = float(
                np.clip(
                    rng.normal(
                        loc=expected_payment_ratio,
                        scale=0.08,
                    ),
                    0.0,
                    1.0,
                )
            )

            amount_paid = amount_due * payment_ratio

            if stress < 0.45:
                days_past_due = 0
            else:
                days_past_due = max(
                    0,
                    int(
                        rng.normal(
                            loc=45 * stress,
                            scale=8,
                        )
                    ),
                )

            records.append(
                {
                    "account_id": account_id,
                    "date": date,
                    "account_type": account_type,
                    "balance": round(balance, 2),
                    "cash_inflow": round(cash_inflow, 2),
                    "cash_outflow": round(cash_outflow, 2),
                    "transaction_count": transaction_count,
                    "credit_utilization": round(
                        credit_utilization,
                        4,
                    ),
                    "amount_due": round(amount_due, 2),
                    "amount_paid": round(amount_paid, 2),
                    "payment_ratio": round(payment_ratio, 4),
                    "days_past_due": days_past_due,
                    "risk_state": risk_state,
                    "risk_state_name": RISK_STATE_NAMES[risk_state],
                }
            )

    return pd.DataFrame(records)


def generate_future_observations(
    latest_accounts: pd.DataFrame,
    *,
    n_days: int = 14,
    random_seed: int = 84,
) -> pd.DataFrame:
    """Generate label-free future rows using the latest observed account state.

    This is a reproducible *approximate continuation*: the original simulator's
    latent stress and credit limits were not persisted. Estimating starting
    values from observed fields does not recover those latent quantities.

    No risk-state labels are returned or used by inference.
    """
    if n_days < 1:
        raise ValueError("n_days must be positive")
    required = {
        "account_id", "date", "account_type", "balance",
        "credit_utilization", "payment_ratio", "days_past_due",
    }
    missing = required - set(latest_accounts.columns)
    if missing:
        raise ValueError(f"Missing latest-account columns: {sorted(missing)}")
    if latest_accounts.empty:
        raise ValueError("latest_accounts cannot be empty")
    if latest_accounts["account_id"].duplicated().any():
        raise ValueError("latest_accounts must contain one row per account")
    rng = np.random.default_rng(random_seed)
    records = []
    latest = latest_accounts.sort_values("account_id")
    for row in latest.itertuples(index=False):
        account_id = row.account_id
        date = pd.Timestamp(row.date)
        if pd.isna(date):
            raise ValueError("Account date cannot be missing")
        account_type = row.account_type
        balance = float(row.balance)
        if not np.isfinite(balance):
            raise ValueError(f"Missing balance for {account_id}")
        utilization = float(row.credit_utilization)
        payment = float(row.payment_ratio)
        past_due = float(row.days_past_due)
        if not all(map(np.isfinite, (utilization, payment, past_due))):
            raise ValueError(f"Missing financial state for {account_id}")
        # Inferring the unobserved stress level from contemporaneous signals.
        stress = float(np.clip(
            0.45 * utilization + 0.45 * (1 - payment)
            + 0.10 * min(past_due / 45, 1.0),
            0.0, 1.0,
        ))
        baseline_stress = stress
        # The real credit limit is latent; using a plausible stable proxy.
        credit_limit = max(3000.0, float(getattr(row, "amount_due", 100.0)) * 30)
        # Generating consecutive future dates after the last observed date.
        future_dates = pd.date_range(
            start=date,
            periods=n_days + 1,
            freq="D",
        )[1:]
        for day in range(1, n_days + 1):
            shock = rng.normal(0, 0.05)
            if rng.random() < 0.015:
                shock += rng.uniform(0.15, 0.35)
            stress = float(np.clip(
                0.85 * stress + 0.15 * baseline_stress + shock, 0, 1
            ))
            cash_inflow = max(0.0, rng.normal(180 * (1 - 0.45 * stress), 60))
            cash_outflow = max(0.0, rng.normal(150 * (1 + 0.55 * stress), 55))
            balance = max(0.0, balance + cash_inflow - cash_outflow)
            transactions = max(0, int(rng.poisson(max(2, 10 - 3 * stress))))
            utilization = float(np.clip(rng.normal(0.25 + 0.65 * stress, 0.08), 0, 1))
            due = max(20.0, credit_limit * utilization * rng.uniform(0.015, 0.040))
            expected_payment = float(np.clip(1.05 - 0.90 * stress, 0.05, 1))
            payment = float(np.clip(rng.normal(expected_payment, 0.08), 0, 1))
            paid = due * payment
            past_due = (0 if stress < 0.45 else max(0, int(rng.normal(45 * stress, 8))))
            records.append(
                {
                    "account_id": account_id,
                    "date": future_dates[day - 1],
                    "account_type": account_type,
                    "balance": round(balance, 2),
                    "cash_inflow": round(cash_inflow, 2),
                    "cash_outflow": round(cash_outflow, 2),
                    "transaction_count": transactions,
                    "credit_utilization": round(utilization, 4),
                    "amount_due": round(due, 2),
                    "amount_paid": round(paid, 2),
                    "payment_ratio": round(payment, 4),
                    "days_past_due": int(past_due),
                }
            )
    return pd.DataFrame.from_records(records)
