from unittest.mock import Mock

from finance_ml.pipelines.silver_pipeline import run_silver_pipeline


def test_run_silver_pipeline_loads_transforms_and_persists(
    monkeypatch,
) -> None:
    """
    Checking that the Silver pipeline loads Bronze data and persists Silver.
    """
    bronze_df = Mock()
    silver_df = Mock()

    spark = Mock()
    spark.table.return_value = bronze_df

    build_silver_mock = Mock(return_value=silver_df)

    monkeypatch.setattr(
        "finance_ml.pipelines.silver_pipeline.build_silver_dataframe",
        build_silver_mock,
    )

    result = run_silver_pipeline(
        spark,
        source_table="bronze_table",
        target_table="silver_table",
    )

    spark.table.assert_called_once_with("bronze_table")

    build_silver_mock.assert_called_once_with(bronze_df)

    silver_df.write.mode.assert_called_once_with("overwrite")

    silver_df.write.mode.return_value.saveAsTable.assert_called_once_with(
        "silver_table"
    )

    assert result is silver_df
