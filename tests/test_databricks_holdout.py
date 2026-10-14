"""Verifying model-bound, read-only Databricks account/date selection."""
from __future__ import annotations

import pandas as pd
import pytest

from finance_ml.data.databricks_holdout import (
    DATE_GOLD, HISTORY_GOLD, NEW_GOLD, DATE_TRUTH, NEW_TRUTH,
    build_select, cohort, gold_query_plan, truth_query_plan,
    validate_uc_request, execute_queries,
)


def request(accounts=("521",), start="2026-03-10", end="2026-03-12", **kwargs):
    return validate_uc_request(account_ids=accounts, start_date=start, end_date=end, **kwargs)


def test_five_cohort_cells():
    assert cohort("ACC_00521", pd.Timestamp("2026-02-01").date()) == "development_period"
    assert cohort("ACC_00521", pd.Timestamp("2026-03-02").date()) == "date_holdout"
    assert cohort("ACC_02001", pd.Timestamp("2026-02-01").date()) == "account_holdout"
    assert cohort("ACC_02001", pd.Timestamp("2026-03-02").date()) == "joint_holdout"


def test_single_known_date_selects_only_date_gold():
    plans = gold_query_plan(request())
    assert [t for t, _ in plans] == [DATE_GOLD]
    assert "'ACC_00521'" in plans[0][1]
    assert "2026-03-04" in plans[0][1]
    assert "2026-03-13" in plans[0][1]
    assert "ORDER BY account_id, date" in plans[0][1]


def test_crossing_cutoff_queries_development_and_date_gold():
    plans = gold_query_plan(request(start="2026-02-28", end="2026-03-04"))
    assert [t for t, _ in plans] == [HISTORY_GOLD, DATE_GOLD]
    assert "2026-03-02" in plans[1][1]


def test_new_account_is_always_new_gold():
    plans = gold_query_plan(request(accounts=("2001",), start="2026-01-01", end="2026-04-30"))
    assert len(plans) == 1 and plans[0][0] == NEW_GOLD


def test_mixed_account_sources_and_optional_truth():
    req = request(accounts=("1", "2001"), start="2026-02-28", end="2026-03-03", evaluate=True)
    assert [t for t, _ in gold_query_plan(req)] == [HISTORY_GOLD, DATE_GOLD, NEW_GOLD]
    assert [t for t, _ in truth_query_plan(req)] == [DATE_TRUTH, NEW_TRUTH]
    assert "2026-03-02" in truth_query_plan(req)[0][1]


def test_development_only_has_no_truth_queries():
    req = request(start="2026-02-01", end="2026-02-02", evaluate=True)
    assert truth_query_plan(req) == []

@pytest.mark.parametrize("kwargs", [
    dict(accounts=("3001",)), dict(accounts=()), dict(accounts=("01", "ACC_00001")),
    dict(start="2025-12-31"), dict(end="2026-05-01"),
    dict(start="2026-03-20", end="2026-03-10"),
    dict(query_method="spark"), dict(model_version="2"),
    dict(model_name="finance_ml.models.some_other_model"),
    dict(accounts=("1", "2", "3"), smoke=True),
    dict(start="2026-03-01", end="2026-03-09", smoke=True),
])
def test_unsupported_input_fails_before_network(kwargs):
    with pytest.raises(ValueError):
        request(**kwargs)


def test_sql_is_allowlisted_and_bounds_are_inclusive():
    with pytest.raises(ValueError):
        build_select("finance_ml.gold.account_features; DROP TABLE x", ["ACC_00001"],
                     pd.Timestamp("2026-03-02").date(), pd.Timestamp("2026-03-04").date())


def test_duplicate_source_rows_fail_closed():
    frame = pd.DataFrame([{"account_id": "ACC_00001", "date": "2026-03-05"}])
    with pytest.raises(ValueError, match="duplicated"):
        execute_queries([(HISTORY_GOLD, "a"), (DATE_GOLD, "b")], lambda sql: frame)


def test_utc_aware_databricks_dates_are_normalized_to_naive_utc():
    frame = pd.DataFrame({
        "account_id": ["ACC_00521"] * 2,
        "date": pd.date_range("2026-03-10", periods=2, tz="Etc/UTC"),
    })
    result = execute_queries([(DATE_GOLD, "SELECT ...")], lambda _: frame)
    assert str(result["date"].dtype) == "datetime64[ns]"
    assert result["date"].tolist() == list(pd.date_range("2026-03-10", periods=2))


def test_utc_aware_and_naive_sources_merge_without_date_type_conflict():
    aware = pd.DataFrame({"account_id": ["ACC_00521"],
                          "date": [pd.Timestamp("2026-03-10", tz="Etc/UTC")]})
    naive = pd.DataFrame({"account_id": ["ACC_00521"],
                          "date": [pd.Timestamp("2026-03-11")]})
    result = execute_queries([(DATE_GOLD, "aware"), (HISTORY_GOLD, "naive")],
                             lambda sql: aware if sql == "aware" else naive)
    requested = pd.DataFrame({"account_id": ["ACC_00521"] * 2,
                              "date": pd.date_range("2026-03-10", periods=2)})
    matched = requested.merge(result, on=["account_id", "date"], validate="one_to_one")
    assert len(matched) == 2


def test_non_midnight_utc_dates_are_rejected_without_truncation():
    data = pd.DataFrame({"account_id": ["ACC_00521"],
                         "date": [pd.Timestamp("2026-03-10T01:00:00Z")]})
    with pytest.raises(ValueError, match="midnight UTC"):
        execute_queries([(DATE_GOLD, "SELECT ...")], lambda _: data)
