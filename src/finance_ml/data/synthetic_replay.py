"""Replay requested simulator accounts without building the full cohort in memory.

Following the original simulator's account/RNG ordering, including its separate
new-account random stream and fixed baseline horizon. Changing the requested
end date cannot change earlier simulated records or another account's history.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date

import numpy as np
import pandas as pd

from finance_ml.data.generate_synthetic import _day, _start_account
from finance_ml.data.synthetic_provenance import SyntheticProvenance

LABELS = ("risk_state", "risk_state_name")


def replay_selected_accounts(profile: SyntheticProvenance,
                             account_ids: tuple[str, ...], through: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (predictor observations, separate simulator truth) through date.

    Known-account historical records use the original shared stream. Future
    records resume their exact end-of-history per-account RNG snapshots. New
    accounts use the original independent cohort stream, always consuming the
    fixed profile anchor for predecessor accounts before individual extension.
    """
    if through < profile.start:
        raise ValueError("Requested end precedes simulator start")
    end = pd.Timestamp(through)
    start = pd.Timestamp(profile.start)
    history_dates = pd.date_range(start, periods=profile.historical_days, freq="D")
    baseline_dates = pd.date_range(start, periods=profile.anchor_days, freq="D")
    selected = {int(x.split("_")[1]) for x in account_ids}
    known = {x for x in selected if x <= profile.development_accounts}
    new = selected - known
    rows: list[dict] = []

    # Preserving the exact original history RNG progression through ALL known
    # accounts, including unselected accounts, without retaining their rows.
    rng = np.random.default_rng(profile.seed)
    snapshots = {}
    for number in range(1, profile.development_accounts + 1):
        state = _start_account(number, rng)
        for day in history_dates:
            record = _day(state, day, rng)
            if number in known and day <= end:
                rows.append(record)
        if number in known:
            snapshots[number] = (deepcopy(state), deepcopy(rng.bit_generator.state))

    for number, (state, rng_state) in snapshots.items():
        resumed = np.random.default_rng()
        resumed.bit_generator.state = deepcopy(rng_state)
        # Continuing from the day after the development period.
        for day in pd.date_range(
            start=profile.development_end,
            end=end,
            freq="D",
        )[1:]:
            rows.append(_day(state, day, resumed))

    if new:
        new_rng = np.random.default_rng(
            np.random.SeedSequence([profile.seed, profile.development_accounts, 2026])
        )
        for number in range(profile.development_accounts + 1, max(new) + 1):
            state = _start_account(number, new_rng)
            for day in baseline_dates:
                record = _day(state, day, new_rng)
                if number in new and day <= end:
                    rows.append(record)
            if number in new and end > baseline_dates[-1]:
                # The new account's extension runs from its own saved state,
                # never consuming the stream used by following new accounts.
                resumed = np.random.default_rng()
                resumed.bit_generator.state = deepcopy(new_rng.bit_generator.state)
                # Continuing from the day after the new-account baseline.
                for day in pd.date_range(
                    start=baseline_dates[-1],
                    end=end,
                    freq="D",
                )[1:]:
                    rows.append(_day(state, day, resumed))

    if not rows:
        raise ValueError("No simulator observations were generated")
    data = pd.DataFrame(rows).sort_values(["account_id", "date"]).reset_index(drop=True)
    if data.duplicated(["account_id", "date"]).any():
        raise AssertionError("Simulator replay produced duplicate keys")
    truth = data[["account_id", "date", *LABELS]].copy()
    observations = data.drop(columns=list(LABELS))
    return observations, truth
