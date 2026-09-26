-- Counting Bronze records.
SELECT COUNT(*) AS row_count
FROM finance_ml.bronze.account_events_raw;


-- Summarizing target-class distribution.
SELECT
    risk_state,
    risk_state_name,
    COUNT(*) AS n_rows
FROM finance_ml.bronze.account_events_raw
GROUP BY
    risk_state,
    risk_state_name
ORDER BY risk_state;


-- Identifying duplicate account-date records.
SELECT
    account_id,
    date,
    COUNT(*) AS n_records
FROM finance_ml.bronze.account_events_raw
GROUP BY
    account_id,
    date
HAVING COUNT(*) > 1;


-- Counting selected invalid values.
SELECT
    SUM(
        CASE
            WHEN cash_outflow < 0 THEN 1
            ELSE 0
        END
    ) AS negative_cash_outflows,

    SUM(
        CASE
            WHEN credit_utilization > 1 THEN 1
            ELSE 0
        END
    ) AS invalid_credit_utilization

FROM finance_ml.bronze.account_events_raw;