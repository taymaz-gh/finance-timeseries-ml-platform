# Data Dictionary

This project simulates daily financial-account time-series data for a sequence-classification task.

Each row represents one account on one day.

## Identifier and time columns

| Column | Type | Meaning |
|---|---|---|
| `account_id` | string | Unique identifier of the financial account. |
| `date` | date | Date of the daily observation. |
| `account_type` | categorical | Simulated account category: `checking`, `credit`, or `business`. |

The pair:

`(account_id, date)`

is intended to uniquely identify one daily account observation.

## Financial variables

| Column | Type | Meaning |
|---|---|---|
| `balance` | float | Current amount of money available in the account. |
| `cash_inflow` | float | Money entering the account during the day. |
| `cash_outflow` | float | Money leaving the account during the day. |
| `transaction_count` | integer | Number of transactions during the day. |
| `credit_utilization` | float | Fraction of the available credit limit currently being used. |
| `amount_due` | float | Amount expected to be paid by the customer. |
| `amount_paid` | float | Amount actually paid by the customer. |
| `payment_ratio` | float | Ratio of `amount_paid` to `amount_due`. |
| `days_past_due` | integer | Number of days that a required payment is overdue. |

## Target variables

| Column | Type | Meaning |
|---|---|---|
| `risk_state` | integer | Numeric risk-state label used for classification. |
| `risk_state_name` | categorical | Human-readable name of the risk state. |

The synthetic risk states are:

| Label | State | Interpretation |
|---:|---|---|
| 0 | `normal` | Financially stable behavior with no significant stress indicators. |
| 1 | `elevated_risk` | Early warning signs such as worsening balance or increasing credit usage. |
| 2 | `liquidity_stress` | Increasing difficulty meeting short-term financial obligations. |
| 3 | `delinquency_risk` | Strong signs of overdue or missed-payment behavior. |

These labels and thresholds are simulation definitions for this project and are not regulatory banking classifications.

## Synthetic latent process

The observable financial variables are influenced by an internally simulated latent financial-stress variable.

Higher stress generally leads to patterns such as:

- lower cash inflow,
- higher cash outflow,
- lower payment ratios,
- higher credit utilization,
- more days past due,
- declining balances.

The latent stress variable itself is not included as a model input.

## Intentional Bronze-layer data-quality issues

Before ingestion into the Bronze layer, a controlled corruption step introduces:

- missing numeric values,
- missing account categories,
- duplicate `(account_id, date)` records,
- negative `cash_outflow` values,
- `credit_utilization` values above `1.0`.

These issues are introduced deliberately so that the later Silver-layer cleaning pipeline can validate and correct them.