from finance_ml.data.build_silver import (
    build_silver_dataframe,
)


def test_build_silver_function_is_importable() -> None:
    """Checking that the Silver builder can be imported locally."""

    assert callable(build_silver_dataframe)
