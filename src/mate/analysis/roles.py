from __future__ import annotations

import json
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


Position = Tuple[int, int]


def fixed_maze_walls(grid_size: int) -> np.ndarray:
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    if grid_size >= 8:
        mid = grid_size // 2
        grid[2 : grid_size - 2, mid] = 1
        grid[mid, 2 : grid_size - 2] = 1
        grid[2, mid] = 0
        grid[grid_size - 3, mid] = 0
        grid[mid, 3] = 0
        grid[mid, grid_size - 4] = 0
    return grid


def parse_json_list(value) -> list:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return json.loads(value)


def classify_topology(pos: Position, walls: np.ndarray) -> str:
    row, col = pos
    open_neighbors = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < walls.shape[0] and 0 <= nc < walls.shape[1] and walls[nr, nc] == 0:
            open_neighbors += 1
    if open_neighbors >= 3:
        return "junction"
    if open_neighbors == 2:
        return "corridor"
    return "dead_end"


def marker_distances_vectorised(
    episode_rollout: pd.DataFrame,
    markers: Iterable[Position],
) -> np.ndarray:
    """
    Compute, in a fully vectorised way, the Manhattan distance from the agent
    to the nearest marker at each timestep in the episode.
    """
    markers = list(markers)
    T = len(episode_rollout)
    if T == 0:
        return np.asarray([], dtype=float)
    if not markers:
        # No markers: distance is effectively infinite everywhere
        return np.full(T, np.inf, dtype=float)

    # Agent positions: shape (T, 2)
    agent_xy = episode_rollout[["agent_x", "agent_y"]].to_numpy(dtype=np.int16)

    # Marker positions: shape (M, 2)
    markers_xy = np.asarray(markers, dtype=np.int16)  # (M, 2)

    # Broadcast to (T, M, 2), then L1 distance
    diff = np.abs(agent_xy[:, None, :] - markers_xy[None, :, :])  # (T, M, 2)
    dists = diff.sum(axis=2)  # (T, M)

    # Nearest marker per timestep: shape (T,)
    return dists.min(axis=1)


def classify_marker_motion(
    episode_rollout: pd.DataFrame,
    markers: Iterable[Position],
) -> Counter:
    """
    Classify motion relative to markers using vectorised distances.

    For each consecutive pair of timesteps:
      - if distance decreases → moves_toward_marker
      - if distance increases → moves_away_from_marker
      - if distance unchanged → ignores
    """
    markers = list(markers)
    counts: Counter = Counter()

    # Preserve original behaviour: if no markers or <2 timesteps, everything is "ignores"
    if not markers or len(episode_rollout) < 2:
        counts["ignores"] += max(len(episode_rollout), 1)
        return counts

    ordered = episode_rollout.sort_values("t")
    distances = marker_distances_vectorised(ordered, markers)

    for before, after in zip(distances[:-1], distances[1:]):
        if after < before:
            counts["moves_toward_marker"] += 1
        elif after > before:
            counts["moves_away_from_marker"] += 1
        else:
            counts["ignores"] += 1

    return counts


def classify_marker_roles(rollouts: pd.DataFrame, markers: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if markers.empty or rollouts.empty:
        return pd.DataFrame(rows)

    group_cols = ["run_id", "condition", "seed", "intervention", "episode_index"]

    # Pre-merge once: each marker row gets all its rollout rows
    merged = markers.merge(
        rollouts,
        on=group_cols,
        how="left",
        suffixes=("_marker", "_rollout"),
    )

    # Group by each episode (one marker map per group)
    for keys, group in merged.groupby(group_cols):
        # Use the first row of this group as the marker row
        marker_row = group.iloc[0]

        # Parse marker map (12×12 grid of 0/1)
        marker_map = np.asarray(
    parse_json_list(
        marker_row.get("marker_map_marker", marker_row.get("marker_map"))
    ),
    dtype=np.int8
)

        if marker_map.size == 0:
            continue

        grid_size = marker_map.shape[0]
        walls = fixed_maze_walls(grid_size)

        # Marker positions as (row, col) pairs
        marker_positions = [tuple(pos) for pos in np.argwhere(marker_map == 1).tolist()]

        # Topology roles
        topology_counts = Counter(classify_topology(pos, walls) for pos in marker_positions)

        # All rollout rows for this episode
        episode_rollout = group.sort_values("t")
        motion_counts = classify_marker_motion(episode_rollout, marker_positions)

        total_topology = sum(topology_counts.values()) or 1
        total_motion = sum(motion_counts.values()) or 1
        base = {col: marker_row[col] for col in group_cols}

        for role, count in topology_counts.items():
            rows.append(
                {
                    **base,
                    "role": role,
                    "count": count,
                    "frequency": count / total_topology,
                }
            )
        for role, count in motion_counts.items():
            rows.append(
                {
                    **base,
                    "role": role,
                    "count": count,
                    "frequency": count / total_motion,
                }
            )

    return pd.DataFrame(rows)


def role_entropy(role_counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if role_counts.empty:
        return pd.DataFrame(rows)

    group_cols = ["run_id", "condition", "seed", "intervention"]

    for keys, group in role_counts.groupby(group_cols):
        probs = group.groupby("role")["count"].sum().to_numpy(dtype=float)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        entropy = -float(np.sum([p * np.log(p) for p in probs if p > 0]))
        rows.append({**dict(zip(group_cols, keys)), "role_entropy": entropy})

    return pd.DataFrame(rows)
