-- Validating the persisted Gold feature table.

-- Checking total row count.
SELECT COUNT(*) AS n_rows
FROM finance_ml.gold.account_features;


-- Checking duplicate account-date keys.
SELECT COUNT(*) AS duplicate_keys
FROM (
    SELECT
        account_id,
        date,
        COUNT(*) AS n_records
    FROM finance_ml.gold.account_features
    GROUP BY
        account_id,
        date
    HAVING COUNT(*) > 1
) AS duplicate_groups;


-- Checking missing target labels.
SELECT
    SUM(CASE WHEN risk_state IS NULL THEN 1 ELSE 0 END)
        AS missing_risk_state,
    SUM(CASE WHEN risk_state_name IS NULL THEN 1 ELSE 0 END)
        AS missing_risk_state_name
FROM finance_ml.gold.account_features;


-- Checking that lag features are missing only where expected.
SELECT
    SUM(CASE WHEN previous_balance IS NULL THEN 1 ELSE 0 END)
        AS missing_previous_balance,
    SUM(CASE WHEN previous_credit_utilization IS NULL THEN 1 ELSE 0 END)
        AS missing_previous_credit_utilization,
    SUM(CASE WHEN previous_payment_ratio IS NULL THEN 1 ELSE 0 END)
        AS missing_previous_payment_ratio,
    SUM(CASE WHEN previous_days_past_due IS NULL THEN 1 ELSE 0 END)
        AS missing_previous_days_past_due
FROM finance_ml.gold.account_features;


-- Checking missing derived features.
SELECT
    SUM(CASE WHEN net_cash_flow IS NULL THEN 1 ELSE 0 END)
        AS missing_net_cash_flow,
    SUM(CASE WHEN payment_gap IS NULL THEN 1 ELSE 0 END)
        AS missing_payment_gap
FROM finance_ml.gold.account_features;


-- Checking target-state distribution.
SELECT
    risk_state,
    risk_state_name,
    COUNT(*) AS n_rows
FROM finance_ml.gold.account_features
GROUP BY
    risk_state,
    risk_state_name
ORDER BY
    risk_state;