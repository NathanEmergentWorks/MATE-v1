from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


def estimate_mutual_information(
    df: pd.DataFrame,
    state_col: str = "state_hash",
    action_col: str = "action",
) -> float:
    data = df[[state_col, action_col]].dropna()
    if data.empty:
        return 0.0

    joint = data.groupby([state_col, action_col]).size().astype(float)
    joint = joint / joint.sum()
    state_probs = data.groupby(state_col).size().astype(float)
    state_probs = state_probs / state_probs.sum()
    action_probs = data.groupby(action_col).size().astype(float)
    action_probs = action_probs / action_probs.sum()

    mi = 0.0
    for (state, action), p_sa in joint.items():
        p_s = state_probs.loc[state]
        p_a = action_probs.loc[action]
        mi += p_sa * np.log(p_sa / (p_s * p_a))
    return float(mi)


def add_discrete_state_hash(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "state_hash" in out.columns and out["state_hash"].notna().any():
        return out
    state_cols = ["agent_x", "agent_y", "goal_x", "goal_y", "marker_map"]
    out["state_hash"] = out[state_cols].astype(str).agg("|".join, axis=1)
    return out


def mutual_information_by_group(
    rollouts: pd.DataFrame,
    group_cols: Iterable[str] = ("condition", "run_id", "seed", "intervention"),
    state_col: str = "state_hash",
    action_col: str = "action",
) -> pd.DataFrame:
    data = add_discrete_state_hash(rollouts)
    rows: List[dict] = []
    group_cols = [col for col in group_cols if col in data.columns]
    for keys, group in data.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["mi"] = estimate_mutual_information(group, state_col, action_col)
        row["n"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows)
