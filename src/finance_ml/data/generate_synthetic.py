"""Generating reproducible synthetic financial account time series."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import pandas as pd

RISK_STATE_NAMES = {0: "normal", 1: "elevated_risk", 2: "liquidity_stress", 3: "delinquency_risk"}


@dataclass
class _AccountState:
    account_id: str
    account_type: str
    baseline_stress: float
    stress: float
    balance: float
    credit_limit: float


def _stress_to_state(stress: float) -> int:
    """Converting latent stress to a class."""
    if stress < .35: return 0
    if stress < .55: return 1
    if stress < .75: return 2
    return 3


def _start_account(number: int, rng: np.random.Generator) -> _AccountState:
    """Initializing an account using exactly the historical RNG order."""
    account_type = rng.choice(["checking", "credit", "business"], p=[.50, .30, .20])
    baseline = float(rng.choice([.15, .40, .65, .85], p=[.55, .25, .15, .05]))
    stress = float(np.clip(rng.normal(loc=baseline, scale=.08), 0., 1.))
    balance = float(rng.lognormal(mean=np.log(5000), sigma=.45))
    credit_limit = float(rng.uniform(3000, 15000))
    return _AccountState(f"ACC_{number:05d}", account_type, baseline, stress, balance, credit_limit)


def _day(account: _AccountState, date: pd.Timestamp, rng: np.random.Generator) -> dict:
    """Advancing one day using the original equations and random draw order."""
    shock = rng.normal(loc=0.0, scale=0.05)
    if rng.random() < 0.015:
        shock += rng.uniform(0.15, 0.35)
    account.stress = float(np.clip(
        .85 * account.stress + .15 * account.baseline_stress + shock, 0., 1.
    ))
    stress = account.stress
    state = _stress_to_state(stress)
    inflow = max(0.0, rng.normal(loc=180.0 * (1.0 - .45 * stress), scale=60.0))
    outflow = max(0.0, rng.normal(loc=150.0 * (1.0 + .55 * stress), scale=55.0))
    account.balance = max(0.0, account.balance + inflow - outflow)
    transactions = max(0, int(rng.poisson(lam=max(2.0, 10.0 - 3.0 * stress))))
    utilization = float(np.clip(rng.normal(loc=.25 + .65 * stress, scale=.08), 0., 1.))
    due = max(20.0, account.credit_limit * utilization * rng.uniform(.015, .040))
    expected_payment = np.clip(1.05 - .90 * stress, .05, 1.0)
    payment = float(np.clip(rng.normal(loc=expected_payment, scale=.08), 0., 1.))
    paid = due * payment
    past_due = 0 if stress < .45 else max(0, int(rng.normal(loc=45 * stress, scale=8)))
    return {
        "account_id": account.account_id, "date": date, "account_type": account.account_type,
        "balance": round(account.balance, 2), "cash_inflow": round(inflow, 2),
        "cash_outflow": round(outflow, 2), "transaction_count": transactions,
        "credit_utilization": round(utilization, 4), "amount_due": round(due, 2),
        "amount_paid": round(paid, 2), "payment_ratio": round(payment, 4),
        "days_past_due": past_due, "risk_state": state,
        "risk_state_name": RISK_STATE_NAMES[state],
    }


def _validate_positive(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def generate_financial_timeseries(
    n_accounts: int = 2000, n_days: int = 60,
    start_date: str = "2026-01-01", random_seed: int = 42,
) -> pd.DataFrame:
    """Generating original historical data; preserving the legacy RNG order."""
    _validate_positive(n_accounts, "n_accounts")
    _validate_positive(n_days, "n_days")
    rng = np.random.default_rng(random_seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    records = []
    for number in range(1, n_accounts + 1):
        account = _start_account(number, rng)
        for date in dates:
            records.append(_day(account, date, rng))
    return pd.DataFrame(records)


def generate_holdout_timeseries(
    *, n_existing_accounts: int = 2000, historical_days: int = 60,
    holdout_days: int = 14, n_new_accounts: int = 500,
    start_date: str = "2026-01-01", random_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Replaying history, resuming known accounts, and simulating new accounts.

    The three outputs contain synthetic labels as ground truth. Downstream
    inference must drop risk_state and risk_state_name from its input features.
    Each continuation restores the account's latent state and the RNG state
    immediately after its original history, without estimated hidden values.
    """
    _validate_positive(n_existing_accounts, "n_existing_accounts")
    _validate_positive(historical_days, "historical_days")
    _validate_positive(holdout_days, "holdout_days")
    if (isinstance(n_new_accounts, (bool, np.bool_)) or
        not isinstance(n_new_accounts, (int, np.integer)) or n_new_accounts < 0):
        raise ValueError("n_new_accounts must be a non-negative integer")

    history_dates = pd.date_range(start=start_date, periods=historical_days, freq="D")
    future_dates = pd.date_range(
        start=history_dates[-1] + pd.Timedelta(days=1), periods=holdout_days, freq="D"
    )
    rng = np.random.default_rng(random_seed)
    history_records = []
    snapshots = []
    for number in range(1, n_existing_accounts + 1):
        account = _start_account(number, rng)
        for date in history_dates:
            history_records.append(_day(account, date, rng))
        snapshots.append((deepcopy(account), deepcopy(rng.bit_generator.state)))

    # Separating the new-account random stream from the historical replay.
    new_rng = np.random.default_rng(np.random.SeedSequence(
        [random_seed, n_existing_accounts, 2026]
    ))
    new_records = []
    for number in range(n_existing_accounts + 1, n_existing_accounts + n_new_accounts + 1):
        account = _start_account(number, new_rng)
        for date in history_dates.append(future_dates):
            new_records.append(_day(account, date, new_rng))

    future_records = []
    for account, rng_state in snapshots:
        continued_rng = np.random.default_rng()
        continued_rng.bit_generator.state = deepcopy(rng_state)
        for date in future_dates:
            future_records.append(_day(account, date, continued_rng))

    columns = list(history_records[0])
    return {
        "historical": pd.DataFrame(history_records, columns=columns),
        "temporal_holdout": pd.DataFrame(future_records, columns=columns),
        "account_holdout": pd.DataFrame(new_records, columns=columns),
    }


def generate_future_observations(
    latest_accounts: pd.DataFrame, *, n_days: int = 14, random_seed: int = 84,
) -> pd.DataFrame:
    """Legacy approximate generator; retained until pipeline migration.

    The holdout redesign must use generate_holdout_timeseries instead.
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
        stress = float(np.clip(
            0.45 * utilization + 0.45 * (1 - payment)
            + 0.10 * min(past_due / 45, 1.0), 0.0, 1.0,
        ))
        baseline_stress = stress
        credit_limit = max(3000.0, float(getattr(row, "amount_due", 100.0)) * 30)
        future_dates = pd.date_range(start=date, periods=n_days + 1, freq="D")[1:]
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
            records.append({
                "account_id": account_id, "date": future_dates[day - 1],
                "account_type": account_type, "balance": round(balance, 2),
                "cash_inflow": round(cash_inflow, 2),
                "cash_outflow": round(cash_outflow, 2),
                "transaction_count": transactions,
                "credit_utilization": round(utilization, 4),
                "amount_due": round(due, 2), "amount_paid": round(paid, 2),
                "payment_ratio": round(payment, 4), "days_past_due": int(past_due),
            })
    return pd.DataFrame.from_records(records)
