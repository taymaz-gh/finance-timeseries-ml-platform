# Gold Feature Engineering Rules

The Gold layer converts cleaned Silver account-event data into model-ready
features for downstream machine-learning workflows.

The source table is:

    finance_ml.silver.account_events_clean

The target Gold table is:

    finance_ml.gold.account_features


## Prediction-time causality

All features for day `t` must use only information that is available at or
before day `t`.

Future observations must never be used when constructing features for the
current timestep.

This prevents temporal leakage.


## Target leakage

The target columns are:

- `risk_state`
- `risk_state_name`

The current target values must never be used as model input features.

Future target values must also never be used.

Lagged target values may only be considered later as an explicit modeling
choice and must not be introduced silently.


## Same-day derived features

The following features may be derived from variables available on the same
day:

- `net_cash_flow = cash_inflow - cash_outflow`
- `payment_gap = amount_due - amount_paid`

These features do not use future information.


## Lag features

Lag features use values from earlier timestamps belonging to the same
`account_id`.

Initial lag features:

- previous `balance`
- previous `credit_utilization`
- previous `payment_ratio`
- previous `days_past_due`

For day `t`, these features use values from day `t-1` only.


## Change features

Change features compare the current value with the previous observed value.

Initial change features:

- `balance_change`
- `credit_utilization_change`
- `payment_ratio_change`

For example:

    balance_change_t =
        balance_t - balance_(t-1)


## Rolling features

Rolling features summarize recent historical behavior within the same
account.

Initial rolling features:

- rolling mean of `balance`
- rolling mean of `cash_outflow`
- rolling mean of `payment_ratio`
- rolling maximum of `days_past_due`

The first implementation will use trailing windows that include only the
current and earlier timestamps.

No future rows may enter a rolling window.


## Missing values

Gold feature engineering must preserve the distinction between:

- observed values
- Silver-imputed values
- remaining missing values

Features derived from missing inputs may remain missing unless a specific
Gold-layer rule is defined.

Flow-like variables such as `cash_inflow` and `cash_outflow` must not be
silently invented through forward filling.


## Account-level ordering

All lag, change, and rolling calculations must be performed:

- within the same `account_id`
- ordered by `date`

Rows from different accounts must never influence one another.


## Initial Gold output

The initial Gold table should contain:

- identifiers
- timestamp
- original cleaned predictor columns
- engineered causal features
- target columns for supervised training

The target columns remain in the table for training and evaluation, but
must be excluded from the model input feature matrix.


## Rolling-window horizon

The initial Gold feature set uses a 7-day trailing calendar window.

For each `account_id`, rolling calculations are ordered by `date` and include
all observations whose timestamps fall between day `t-6` and day `t`,
inclusive.

This is implemented with a date-aware Spark range window rather than a
row-count window.

Using a row-count window such as `rowsBetween(-6, 0)` would mean
"the current row plus six previous observations." That is only equivalent to
seven calendar days when every account has exactly one observation on every
day.

Because Silver may contain missing dates after rows with missing targets are
removed, the Gold layer uses calendar-time semantics instead.

The rolling window therefore remains correct even when observations are
irregularly spaced.


This gives a maximum window of seven daily observations:

    t-6, t-5, t-4, t-3, t-2, t-1, t

The window never includes future timestamps.

If fewer than seven earlier observations are available, the rolling
statistic is calculated from the available observations in the trailing
window.

The initial rolling features are:

- `rolling_7d_mean_balance`
- `rolling_7d_mean_cash_outflow`
- `rolling_7d_mean_payment_ratio`
- `rolling_7d_max_days_past_due`


## Lag semantics with missing dates

Lag features represent the most recent previous available observation within
the same `account_id`.

They should not be interpreted as necessarily coming from the previous
calendar day.

For example, if an account has observations on January 8 and January 10 but
not January 9, the January 10 lag feature uses the January 8 observation.

This behavior is intentional and differs from the date-aware 7-day rolling
features, which are defined using elapsed calendar time.

The distinction is important:

lag → previous available observation

7-day rolling feature → actual trailing 7 calendar days


## Validation results

The persisted Gold table was validated after feature generation.

Observed results:

- total rows: 120,000
- duplicate `(account_id, date)` keys: 0
- missing `risk_state`: 0
- missing `risk_state_name`: 0
- missing `previous_balance`: 2,025
- missing `previous_credit_utilization`: 2,023
- missing `previous_payment_ratio`: 2,015
- missing `previous_days_past_due`: 2,000
- missing `net_cash_flow`: 2,623
- missing `payment_gap`: 0

The lag-feature missing counts are consistent with the 2,000 account-first rows,
plus unresolved Silver-layer missing values.

The target distribution remained unchanged:

- `0 normal`: 68,444
- `1 elevated_risk`: 23,167
- `2 liquidity_stress`: 17,929
- `3 delinquency_risk`: 10,460