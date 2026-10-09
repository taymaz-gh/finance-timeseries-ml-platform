"""Testing MLflow LoggedModel selection and validation."""

from __future__ import annotations

import pytest

from types import SimpleNamespace

from finance_ml.models.registry import (
    register_logged_model,
    resolve_logged_model_uri,
)


EXPERIMENT_ID = "test-experiment"
RUN_ID = "test-run"
MODEL_ID = "m-test123"
MODEL_URI = f"models:/{MODEL_ID}"


class FakeMlflowClient:
    """Simulating MLflow responses without a Databricks connection."""

    def __init__(self) -> None:
        self.logged_model = SimpleNamespace(
            model_id=MODEL_ID,
            name="lstm_model",
            source_run_id=RUN_ID,
            status="READY",
        )

    def get_logged_model(self, model_id: str):
        assert model_id == MODEL_ID
        return self.logged_model

    def search_runs(self, **kwargs):
        return [
            SimpleNamespace(
                info=SimpleNamespace(run_id=RUN_ID)
            )
        ]

    def search_logged_models(self, **kwargs):
        assert kwargs["max_results"] <= 50

        class ModelPage(list):
            token = None

        return ModelPage([self.logged_model])


def test_explicit_model_uri() -> None:
    """Selecting an explicitly provided LoggedModel."""
    client = FakeMlflowClient()

    result = resolve_logged_model_uri(
        model_uri=MODEL_URI,
        client=client,
    )

    assert result == MODEL_URI


def test_final_tag_selection() -> None:
    """Selecting the model associated with the final-tagged run."""
    client = FakeMlflowClient()

    result = resolve_logged_model_uri(
        experiment_id=EXPERIMENT_ID,
        use_final_tag=True,
        client=client,
    )

    assert result == MODEL_URI


def test_combined_selection() -> None:
    """Checking agreement between explicit and final-tagged models."""
    client = FakeMlflowClient()

    result = resolve_logged_model_uri(
        experiment_id=EXPERIMENT_ID,
        model_uri=MODEL_URI,
        use_final_tag=True,
        client=client,
    )

    assert result == MODEL_URI


def test_requires_selection_method() -> None:
    """Rejecting calls without a model-selection method."""
    with pytest.raises(
        ValueError,
        match="Provide model_uri or set use_final_tag=True",
    ):
        resolve_logged_model_uri()


def test_final_tag_requires_experiment_id() -> None:
    """Requiring an experiment ID for tag-based selection."""
    with pytest.raises(
        ValueError,
        match="experiment_id is required",
    ):
        resolve_logged_model_uri(
            use_final_tag=True,
        )


def test_rejects_invalid_model_uri() -> None:
    """Rejecting an invalid MLflow LoggedModel URI."""
    with pytest.raises(
        ValueError,
        match="Expected an MLflow LoggedModel URI",
    ):
        resolve_logged_model_uri(
            model_uri="invalid-uri",
            client=FakeMlflowClient(),
        )


def test_rejects_model_that_is_not_ready() -> None:
    """Rejecting a LoggedModel that is not READY."""
    client = FakeMlflowClient()
    client.logged_model.status = "FAILED"

    with pytest.raises(
        ValueError,
        match="not READY",
    ):
        resolve_logged_model_uri(
            model_uri=MODEL_URI,
            client=client,
        )


def test_rejects_inconsistent_selections() -> None:
    """Rejecting disagreement between explicit and tagged models."""
    client = FakeMlflowClient()

    # Making the final-tagged model differ from the explicit model.
    original_search = client.search_logged_models

    def search_different_model(**kwargs):
        page = original_search(**kwargs)
        page[0] = SimpleNamespace(
            model_id="m-different",
            name="lstm_model",
            source_run_id=RUN_ID,
            status="READY",
        )
        return page

    client.search_logged_models = search_different_model

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        resolve_logged_model_uri(
            experiment_id=EXPERIMENT_ID,
            model_uri=MODEL_URI,
            use_final_tag=True,
            client=client,
        )

def test_rejects_missing_final_run() -> None:
    """Rejecting tag-based selection when no final run exists."""
    client = FakeMlflowClient()

    # Simulating an experiment without final-tagged runs.
    client.search_runs = lambda **kwargs: []

    with pytest.raises(
        LookupError,
        match="No run tagged final_model=true",
    ):
        resolve_logged_model_uri(
            experiment_id=EXPERIMENT_ID,
            use_final_tag=True,
            client=client,
        )


def test_rejects_ambiguous_final_models() -> None:
    """Rejecting multiple eligible models for the selected final run."""
    client = FakeMlflowClient()

    def search_duplicate_models(**kwargs):
        class ModelPage(list):
            token = None

        second_model = SimpleNamespace(
            model_id="m-second",
            name="lstm_model",
            source_run_id=RUN_ID,
            status="READY",
        )

        return ModelPage([
            client.logged_model,
            second_model,
        ])

    client.search_logged_models = search_duplicate_models

    with pytest.raises(
        LookupError,
        match="Expected exactly one READY LoggedModel",
    ):
        resolve_logged_model_uri(
            experiment_id=EXPERIMENT_ID,
            use_final_tag=True,
            client=client,
        )

def test_final_model_discovery_across_pages() -> None:
    """Finding the final model on a later MLflow search page."""
    client = FakeMlflowClient()
    requested_tokens = []

    class ModelPage(list):
        def __init__(self, models, token=None):
            super().__init__(models)
            self.token = token

    def search_paginated_models(**kwargs):
        assert kwargs["max_results"] <= 50

        page_token = kwargs.get("page_token")
        requested_tokens.append(page_token)

        if page_token is None:
            # Simulating a first page without the selected model.
            other_model = SimpleNamespace(
                model_id="m-other",
                name="lstm_model",
                source_run_id="other-run",
                status="READY",
            )

            return ModelPage(
                [other_model],
                token="next-page",
            )

        if page_token == "next-page":
            # Returning the selected final model on page two.
            return ModelPage(
                [client.logged_model],
                token=None,
            )

        raise AssertionError(
            f"Unexpected page token: {page_token}"
        )

    client.search_logged_models = search_paginated_models

    result = resolve_logged_model_uri(
        experiment_id=EXPERIMENT_ID,
        use_final_tag=True,
        client=client,
    )

    assert result == MODEL_URI
    assert requested_tokens == [None, "next-page"]


def test_register_logged_model_reuses_existing_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reusing an existing registered version of the same LoggedModel."""
    import mlflow

    registered_name = "finance_ml.models.lstm_risk_classifier"

    existing_version = SimpleNamespace(
        name=registered_name,
        version="1",
        source="some-artifact-location",
        tags={
            "source_logged_model_id": MODEL_ID,
        },
    )

    class VersionPage(list):
        token = None

    class FakeRegistryClient(FakeMlflowClient):
        def search_model_versions(self, **kwargs):
            assert kwargs["filter_string"] == (
                f"name = '{registered_name}'"
            )
            return VersionPage([existing_version])

    def fail_if_registered(*args, **kwargs):
        raise AssertionError(
            "An existing model version must not be registered again."
        )

    monkeypatch.setattr(
        mlflow,
        "register_model",
        fail_if_registered,
    )

    result = register_logged_model(
        model_uri=MODEL_URI,
        registered_model_name=registered_name,
        client=FakeRegistryClient(),
    )

    assert result is existing_version


def test_register_logged_model_creates_new_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registering and tagging a new Unity Catalog model version."""
    import mlflow

    registered_name = "finance_ml.models.lstm_risk_classifier"
    calls = {}

    new_version = SimpleNamespace(
        name=registered_name,
        version="1",
    )

    class VersionPage(list):
        token = None

    class FakeRegistryClient(FakeMlflowClient):
        def search_model_versions(self, **kwargs):
            return VersionPage([])

        def set_model_version_tag(
            self,
            name,
            version,
            key,
            value,
        ):
            calls["tag"] = {
                "name": name,
                "version": version,
                "key": key,
                "value": value,
            }

    def fake_register_model(*, model_uri, name):
        calls["registration"] = {
            "model_uri": model_uri,
            "name": name,
        }
        return new_version

    monkeypatch.setattr(
        mlflow,
        "register_model",
        fake_register_model,
    )

    result = register_logged_model(
        model_uri=MODEL_URI,
        registered_model_name=registered_name,
        client=FakeRegistryClient(),
    )

    assert result is new_version

    assert calls["registration"] == {
        "model_uri": MODEL_URI,
        "name": registered_name,
    }

    assert calls["tag"] == {
        "name": registered_name,
        "version": "1",
        "key": "source_logged_model_id",
        "value": MODEL_ID,
    }


def test_register_logged_model_targets_unity_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifying that registration targets the Unity Catalog registry."""
    import mlflow

    registered_name = "finance_ml.models.lstm_risk_classifier"
    configured_uris = []

    class VersionPage(list):
        token = None

    class FakeRegistryClient(FakeMlflowClient):
        def search_model_versions(self, **kwargs):
            return VersionPage([])

        def set_model_version_tag(
            self, name, version, key, value
        ):
            pass

    new_version = SimpleNamespace(
        name=registered_name,
        version="1",
    )

    monkeypatch.setattr(
        mlflow,
        "set_registry_uri",
        lambda uri: configured_uris.append(uri),
    )

    monkeypatch.setattr(
        mlflow,
        "register_model",
        lambda **kwargs: new_version,
    )

    result = register_logged_model(
        model_uri=MODEL_URI,
        registered_model_name=registered_name,
        client=FakeRegistryClient(),
    )

    assert configured_uris == ["databricks-uc"]
    assert result is new_version


@pytest.mark.parametrize(
    "invalid_name",
    [
        "",
        "lstm_risk_classifier",
        "models.lstm_risk_classifier",
        "finance_ml..lstm_risk_classifier",
        ".models.lstm_risk_classifier",
        "finance_ml.models.",
    ],
)
def test_register_logged_model_rejects_invalid_name(
    invalid_name: str,
) -> None:
    """Rejecting malformed Unity Catalog model names."""
    with pytest.raises(
        ValueError,
        match="registered_model_name must use",
    ):
        register_logged_model(
            model_uri=MODEL_URI,
            registered_model_name=invalid_name,
            client=FakeMlflowClient(),
        )