"""Bind explicit synthetic account/date provenance to an existing local bundle.

Legacy Step 24 bundles do not capture training population/seed. Profiles are
USER DECLARATIONS, not a cryptographic proof of the original dataset.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from finance_ml.models.local_bundle import inspect_local_bundle

PROFILE_FILENAME = "synthetic_provenance.json"
ACCOUNT_PATTERN = re.compile(r"^(?:ACC_([0-9]{5})|([0-9]{1,5}))$")


def parse_date(value: str, name: str) -> date:
    """Require exact ISO calendar dates, excluding datetime or timezone inputs."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{name} must be YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid calendar date") from exc


def normalize_account_id(raw: str) -> str:
    """Normalize numeric IDs and ACC_XXXXX to the simulator's identifier."""
    if not isinstance(raw, str) or not ACCOUNT_PATTERN.fullmatch(raw):
        raise ValueError(f"Invalid account ID {raw!r}; use 521 or ACC_00521")
    match = ACCOUNT_PATTERN.fullmatch(raw)
    number = int(match.group(1) or match.group(2))
    if number < 1:
        raise ValueError("Account IDs must be positive")
    return f"ACC_{number:05d}"


@dataclass(frozen=True)
class SyntheticProvenance:
    """User-asserted simulation parameters bound to one checkpoint digest."""
    model_sha256: str
    development_accounts: int
    historical_days: int
    seed: int
    start: date
    observed_through: date
    max_new_accounts: int

    @property
    def development_end(self) -> date:
        return self.start + timedelta(days=self.historical_days - 1)

    @property
    def anchor_days(self) -> int:
        return (self.observed_through - self.start).days + 1

    def as_dict(self) -> dict:
        return {
            "profile_version": 1,
            "provenance_level": "user_declared_unverified",
            "simulator": "finance_ml.data.generate_synthetic:stateful_v1",
            "model_sha256": self.model_sha256,
            "development_accounts": self.development_accounts,
            "historical_days": self.historical_days,
            "seed": self.seed,
            "start_date": self.start.isoformat(),
            "observed_through": self.observed_through.isoformat(),
            "max_new_accounts": self.max_new_accounts,
            "new_account_anchor_days": self.anchor_days,
        }


def validate_profile(data: dict, model_sha256: str) -> SyntheticProvenance:
    """Reject modified model associations and malformed replay parameters."""
    if not isinstance(data, dict):
        raise ValueError("Synthetic provenance must be a JSON object")
    expected = {"profile_version", "provenance_level", "simulator", "model_sha256",
                "development_accounts", "historical_days", "seed", "start_date",
                "observed_through", "max_new_accounts", "new_account_anchor_days"}
    if set(data) != expected:
        raise ValueError("Synthetic provenance has missing or unsupported fields")
    if data["profile_version"] != 1 or type(data["profile_version"]) is not int:
        raise ValueError("Unsupported synthetic provenance version")
    if data["provenance_level"] != "user_declared_unverified":
        raise ValueError("Unsupported or misleading provenance level")
    if data["simulator"] != "finance_ml.data.generate_synthetic:stateful_v1":
        raise ValueError("Unsupported simulator version")
    if data["model_sha256"] != model_sha256:
        raise ValueError("Provenance profile belongs to a different Keras checkpoint")
    for name, low, high in (("development_accounts", 1, 2000),
                            ("historical_days", 1, 365), ("seed", 0, 2**32 - 1),
                            ("max_new_accounts", 0, 1000)):
        v = data[name]
        if type(v) is not int or not low <= v <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    start = parse_date(data["start_date"], "start_date")
    observed = parse_date(data["observed_through"], "observed_through")
    p = SyntheticProvenance(
        model_sha256, data["development_accounts"], data["historical_days"],
        data["seed"], start, observed, data["max_new_accounts"],
    )
    if p.observed_through < p.development_end:
        raise ValueError("observed_through must include the full development period")
    if p.anchor_days > 366:
        raise ValueError("Synthetic anchor exceeds the bounded 366-day limit")
    if data["new_account_anchor_days"] != p.anchor_days or type(data["new_account_anchor_days"]) is not int:
        raise ValueError("New-account replay anchor is inconsistent")
    return p


def create_profile(bundle_dir: str | Path, *, development_accounts: int,
                   historical_days: int, seed: int, start_date: str,
                   observed_through: str, max_new_accounts: int = 1000,
                   profile_path: str | Path | None = None) -> Path:
    """Create a non-overwriting, user-declared sidecar; leave bundle untouched."""
    _, manifest = inspect_local_bundle(bundle_dir)
    data = {
        "profile_version": 1,
        "provenance_level": "user_declared_unverified",
        "simulator": "finance_ml.data.generate_synthetic:stateful_v1",
        "model_sha256": manifest["model"]["sha256"],
        "development_accounts": development_accounts,
        "historical_days": historical_days,
        "seed": seed,
        "start_date": start_date,
        "observed_through": observed_through,
        "max_new_accounts": max_new_accounts,
        "new_account_anchor_days": (parse_date(observed_through, "observed_through") -
                                    parse_date(start_date, "start_date")).days + 1,
    }
    validate_profile(data, manifest["model"]["sha256"])
    path = Path(profile_path) if profile_path is not None else Path(bundle_dir) / PROFILE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return path


def load_profile(bundle_dir: str | Path, profile_path: str | Path | None = None,
                 *, manifest: dict | None = None) -> SyntheticProvenance:
    """Load checkpoint-bound provenance without changing the artifact manifest."""
    if manifest is None:
        _, manifest = inspect_local_bundle(bundle_dir)
    path = Path(profile_path) if profile_path is not None else Path(bundle_dir) / PROFILE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Synthetic provenance profile missing: {path}. "
            "Run `finance-ml profile-init` with your verified training parameters."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Synthetic provenance profile is not valid JSON") from exc
    return validate_profile(data, manifest["model"]["sha256"])
