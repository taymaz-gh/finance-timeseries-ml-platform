"""Defining the verified reference input schema for the 41-feature LSTM."""

from __future__ import annotations


REFERENCE_V1_NUMERIC_COLUMNS = (
    "balance",
    "cash_inflow",
    "cash_outflow",
    "transaction_count",
    "credit_utilization",
    "amount_due",
    "amount_paid",
    "payment_ratio",
    "days_past_due",
    "net_cash_flow",
    "payment_gap",
    "previous_balance",
    "previous_credit_utilization",
    "previous_payment_ratio",
    "previous_days_past_due",
    "balance_change",
    "credit_utilization_change",
    "payment_ratio_change",
    "rolling_7d_mean_balance",
    "rolling_7d_mean_cash_outflow",
    "rolling_7d_mean_payment_ratio",
    "rolling_7d_max_days_past_due",
)

REFERENCE_V1_MISSING_COLUMNS = (
    "balance",
    "cash_inflow",
    "cash_outflow",
    "credit_utilization",
    "payment_ratio",
    "net_cash_flow",
    "previous_balance",
    "previous_credit_utilization",
    "previous_payment_ratio",
    "previous_days_past_due",
    "balance_change",
    "credit_utilization_change",
    "payment_ratio_change",
    "rolling_7d_mean_balance",
    "rolling_7d_mean_cash_outflow",
    "rolling_7d_mean_payment_ratio",
)

REFERENCE_V1_CATEGORY_LEVELS = {
    "account_type": ("business", "checking", "credit"),
}

REFERENCE_V1_FINAL_FEATURE_COLUMNS = (
    *REFERENCE_V1_NUMERIC_COLUMNS,
    *(f"{name}_missing" for name in REFERENCE_V1_MISSING_COLUMNS),
    *(f"account_type_{name}" for name in REFERENCE_V1_CATEGORY_LEVELS["account_type"]),
)
