"""
Discovering and validating MLflow LoggedModels for registration.

Supporting explicit model URIs, final-run tags, and consistency checks
before registering models in Unity Catalog.
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient


def resolve_logged_model_uri(
    *,
    experiment_id: str | None = None,
    model_uri: str | None = None,
    use_final_tag: bool = False,
    model_name: str = "lstm_model",
    client: MlflowClient | None = None,
) -> str:
    """
    Resolve the MLflow URI of the model selected for registration.

    Args:
        experiment_id:
            MLflow experiment ID used for tag-based discovery.

        model_uri:
            Optional explicit MLflow LoggedModel URI.

        use_final_tag:
            Whether to discover the model associated with the
            most recent run tagged final_model=true.

        model_name:
            Expected MLflow LoggedModel name.

        client:
            Optional MLflow client, injectable for testing.

    Returns:
        The validated MLflow LoggedModel URI.

    Raises:
        ValueError:
            If neither selection method is provided, or if
            both methods identify different models.

        LookupError:
            If no suitable final-run model is found or the
            selection is ambiguous.
    """
    if model_uri is None and not use_final_tag:
        raise ValueError(
            "Provide model_uri or set use_final_tag=True."
        )

    if use_final_tag and experiment_id is None:
        raise ValueError(
            "experiment_id is required when use_final_tag=True."
        )

    if client is None:
        client = MlflowClient()

    # Validating the explicitly selected model.
    explicit_model_id = None

    if model_uri is not None:
        prefix = "models:/"

        if not model_uri.startswith(prefix):
            raise ValueError(
                "Expected an MLflow LoggedModel URI: models:/<model_id>."
            )

        explicit_model_id = model_uri[len(prefix):]

        if not explicit_model_id.startswith("m-") or "/" in explicit_model_id:
            raise ValueError(
                "Expected an MLflow 3 LoggedModel ID."
            )

        explicit_model = client.get_logged_model(
            explicit_model_id
        )

        if explicit_model.status != "READY":
            raise ValueError(
                "The explicitly selected model is not READY."
            )

    # Returning the explicitly selected model when no tag is requested.
    if not use_final_tag:
        return f"models:/{explicit_model_id}"

    # Discovering the most recent final-model run.
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.final_model = 'true'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise LookupError(
            "No run tagged final_model=true was found."
        )

    final_run_id = runs[0].info.run_id

    # Searching all logged models from the selected experiment.
    matching_models = []
    page_token = None

    while True:
        page = client.search_logged_models(
            experiment_ids=[experiment_id],
            max_results=50,
            page_token=page_token,
        )

        matching_models.extend(
            model
            for model in page
            if model.source_run_id == final_run_id
            and model.name == model_name
            and model.status == "READY"
        )

        page_token = page.token

        if not page_token:
            break

    if len(matching_models) != 1:
        raise LookupError(
            "Expected exactly one READY LoggedModel named "
            f"{model_name!r} for final run {final_run_id}; "
            f"found {len(matching_models)}."
        )

    selected_model_id = matching_models[0].model_id

    # Checking agreement between explicit and tag-based selections.
    if (
        explicit_model_id is not None
        and explicit_model_id != selected_model_id
    ):
        raise ValueError(
            "Explicit model URI does not match the "
            "model associated with the final-tagged run."
        )

    return f"models:/{selected_model_id}"


def register_logged_model(
    *,
    model_uri: str,
    registered_model_name: str,
    client: MlflowClient | None = None,
):
    """
    Register a READY MLflow LoggedModel in Unity Catalog.

    Reusing an existing registered version when its source LoggedModel
    ID matches the requested source.
    """

    import mlflow

    # Targeting the Unity Catalog model registry.
    mlflow.set_registry_uri("databricks-uc")

    # Validating the three-level Unity Catalog model name.
    name_parts = registered_model_name.split(".")

    if (
        len(name_parts) != 3
        or any(not part.strip() for part in name_parts)
    ):
        raise ValueError(
            "registered_model_name must use catalog.schema.model "
            "with non-empty components."
        )

    if client is None:
        client = MlflowClient(registry_uri="databricks-uc")

    # Validating the source LoggedModel.
    validated_uri = resolve_logged_model_uri(
        model_uri=model_uri,
        client=client,
    )

    source_model_id = validated_uri.removeprefix("models:/")

    # Inspecting existing Unity Catalog model versions.
    page_token = None

    while True:
        versions = client.search_model_versions(
            filter_string=f"name = '{registered_model_name}'",
            page_token=page_token,
        )

        for version in versions:
            if (
                version.tags.get("source_logged_model_id")
                == source_model_id
                or version.source == validated_uri
            ):
                return version

        page_token = versions.token

        if not page_token:
            break

    # Registering only when no matching version exists.
    registered_version = mlflow.register_model(
        model_uri=validated_uri,
        name=registered_model_name,
    )

    # Recording source provenance for future duplicate detection.
    client.set_model_version_tag(
        name=registered_model_name,
        version=registered_version.version,
        key="source_logged_model_id",
        value=source_model_id,
    )

    return registered_version