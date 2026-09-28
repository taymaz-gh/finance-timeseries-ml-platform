from unittest.mock import Mock

from finance_ml.pipelines.gold_pipeline import run_gold_pipeline


def test_run_gold_pipeline_loads_transforms_and_persists(
    monkeypatch,
) -> None:
    """
    Checking that the Gold pipeline loads Silver data and persists Gold.
    """
    silver_df = Mock()
    gold_df = Mock()

    spark = Mock()
    spark.table.return_value = silver_df

    build_gold_mock = Mock(return_value=gold_df)

    monkeypatch.setattr(
        "finance_ml.pipelines.gold_pipeline.build_gold_features",
        build_gold_mock,
    )

    result = run_gold_pipeline(
        spark,
        source_table="silver_table",
        target_table="gold_table",
    )

    spark.table.assert_called_once_with("silver_table")

    build_gold_mock.assert_called_once_with(silver_df)

    gold_df.write.mode.assert_called_once_with("overwrite")

    gold_df.write.mode.return_value.saveAsTable.assert_called_once_with("gold_table")

    assert result is gold_df
