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
