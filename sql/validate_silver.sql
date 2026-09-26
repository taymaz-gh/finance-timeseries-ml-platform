-- Counting Silver records.
SELECT COUNT(*) AS n_rows
FROM finance_ml.silver.account_events_clean;


-- Checking invalid values and missing targets.
SELECT
    COUNT(*) AS n_rows,

    SUM(
        CASE
            WHEN cash_outflow < 0 THEN 1
            ELSE 0
        END
    ) AS negative_cash_outflow,

    SUM(
        CASE
            WHEN credit_utilization > 1 THEN 1
            ELSE 0
        END
    ) AS invalid_credit_utilization,

    SUM(
        CASE
            WHEN risk_state IS NULL
              OR risk_state_name IS NULL
            THEN 1
            ELSE 0
        END
    ) AS missing_targets

FROM finance_ml.silver.account_events_clean;


-- Checking duplicate account-date keys.
SELECT COUNT(*) AS duplicate_keys
FROM (
    SELECT
        account_id,
        date,
        COUNT(*) AS n_records
    FROM finance_ml.silver.account_events_clean
    GROUP BY
        account_id,
        date
    HAVING COUNT(*) > 1
);
