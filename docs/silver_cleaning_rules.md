# Silver Cleaning Rules

The Silver layer converts Bronze account-event data into a cleaned,
validated time-series dataset suitable for downstream feature engineering
and machine-learning workflows.

## Primary key

Each valid Silver row must be uniquely identified by:

- `account_id`
- `date`

Duplicate `(account_id, date)` records are resolved deterministically so
that only one record remains.

## Account type

`account_type` is treated as a static account attribute.

If missing, it is filled from another known record belonging to the same
`account_id`.

If no known account type exists for that account, the value remains
`unknown`.

## Invalid numeric values

The following values are treated as invalid:

- `cash_outflow < 0`
- `credit_utilization > 1`

Invalid values are first converted to missing values before imputation.

## Time-series imputation

Numeric missing values are imputed causally within each account whenever
appropriate.

Causal imputation uses only observations from the current or previous
timestamps and does not use future information.

Forward filling is preferred for temporally persistent variables.

If no earlier valid observation exists, the value remains missing unless
a variable-specific fallback rule is defined.

## Target labels

Rows with missing target labels are removed.

The target columns are:

- `risk_state`
- `risk_state_name`

Target values are never imputed.

## Final Silver requirements

The Silver table must:

- contain one row per `(account_id, date)`
- contain no negative `cash_outflow`
- contain no `credit_utilization > 1`
- contain valid target labels
- preserve temporal ordering within each account



## Column-specific imputation policy

### Causal imputation for state-like variables: Forward-filled within each account

The following state-like variables may be imputed using earlier observations
from the same `account_id`:

- `balance`
- `credit_utilization`
- `payment_ratio`
- `days_past_due`

Imputation must use only values that were originally observed in the source
data.

An imputed value must never be used as the source for imputing a later
missing value. This prevents imputation error from propagating through a
sequence of missing observations.

For example:

    day 1   balance = 1000   observed
    day 2   balance = NULL
    day 3   balance = NULL
    day 4   balance = 1300   observed

Using the most recent earlier observed value:

    day 1   balance = 1000   observed
    day 2   balance = 1000   imputed from observed day 1
    day 3   balance = 1000   imputed from observed day 1
    day 4   balance = 1300   observed

Both day 2 and day 3 refer directly to the original observed value from
day 1. Day 3 is not imputed from the already-imputed value at day 2.

Only past observations are used, so the procedure remains causal and does
not use future information.

### Not forward-filled

The following flow- or event-like variables are not forward-filled:

- `cash_inflow`
- `cash_outflow`
- `transaction_count`
- `amount_due`
- `amount_paid`

Forward-filling these variables could invent transactions or obligations
that did not actually occur.

Any remaining missing values in these columns are preserved as missing at
the Silver stage unless a later feature-engineering rule explicitly handles
them.

### Static attribute

`account_type` is treated as a static attribute of an account.

Missing values are filled using another known `account_type` from the same
`account_id`.

If no known value exists for that account, the value is set to `unknown`.

### Target columns

The target columns:

- `risk_state`
- `risk_state_name`

are never imputed.

Rows with missing target values are removed.