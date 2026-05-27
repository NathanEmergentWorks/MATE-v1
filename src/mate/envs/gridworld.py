from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


Position = Tuple[int, int]


@dataclass
class RewardConfig:
    goal: float = 1.0
    step: float = -0.01
    invalid: float = -0.01


class GridWorldEnv:
    """Minimal Gym-style gridworld for the MATE experiment."""

    EMPTY = 0
    WALL = 1

    ACTIONS: Dict[int, Position] = {
        0: (-1, 0),   # up
        1: (1, 0),    # down
        2: (0, -1),   # left
        3: (0, 1),    # right
    }

    def __init__(
        self,
        grid_size: int = 12,
        view_size: int = 5,
        max_steps: int = 200,
        has_markers: bool = False,
        marker_persistence: str = "none",
        walls_layout: str = "fixed_maze_1",
        reward_goal: float = 1.0,
        reward_step: float = -0.01,
        reward_invalid: float = -0.01,
        reward: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        **_: object,
    ) -> None:
        if view_size % 2 == 0:
            raise ValueError("view_size must be odd so the agent can be centered")
        if marker_persistence not in {"none", "episode"}:
            raise ValueError("marker_persistence must be 'none' or 'episode'")

        self.grid_size = int(grid_size)
        self.view_size = int(view_size)
        self.max_steps = int(max_steps)
        self.has_markers = bool(has_markers)
        self.marker_persistence = marker_persistence
        self.walls_layout = walls_layout

        reward = reward or {}
        self.reward_config = RewardConfig(
            goal=float(reward.get("goal", reward_goal)),
            step=float(reward.get("step", reward_step)),
            invalid=float(reward.get("invalid", reward_invalid)),
        )

        self.rng = np.random.default_rng(seed)
        self.grid = self._make_grid(walls_layout)
        self.markers = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.agent_pos: Position = (0, 0)
        self.goal_pos: Position = (0, 0)
        self.step_count = 0
        self.markers_used = 0
        self.marker_placements: List[Tuple[int, int, int]] = []

    @property
    def num_actions(self) -> int:
        return 5 if self.has_markers else 4

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        channels = 4 if self.has_markers else 3
        return (self.view_size, self.view_size, channels)

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> np.ndarray:
        self.grid = self._make_grid(self.walls_layout)
        self.markers.fill(0)
        self.step_count = 0
        self.markers_used = 0
        self.marker_placements = []

        empties = self._empty_cells()
        chosen = self.rng.choice(len(empties), size=2, replace=False)
        self.agent_pos = empties[int(chosen[0])]
        self.goal_pos = empties[int(chosen[1])]
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"invalid action {action} for num_actions={self.num_actions}")

        self.step_count += 1
        reward = self.reward_config.step

        # Marker placement
        if self.has_markers and action == 4:
            self.markers[self.agent_pos] = 1
            if self.marker_persistence == "none":
                self.markers[self.agent_pos] = 0
            self.markers_used += 1
            self.marker_placements.append((self.step_count - 1, self.agent_pos[0], self.agent_pos[1]))

        # Movement
        elif action in self.ACTIONS:
            dr, dc = self.ACTIONS[action]
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if self._is_wall_or_oob(nr, nc):
                reward = self.reward_config.invalid
            else:
                self.agent_pos = (nr, nc)

        # Termination
        reached_goal = self.agent_pos == self.goal_pos
        if reached_goal:
            reward += self.reward_config.goal

        done = reached_goal or self.step_count >= self.max_steps

        info = {
            "success": int(reached_goal),
            "steps": self.step_count,
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "markers_used": self.markers_used,
            "marker_map": self.markers.copy(),
            "placements": list(self.marker_placements),
        }

        return self._get_obs(), float(reward), bool(done), info

    def marker_map(self) -> np.ndarray:
        return self.markers.copy()

    def set_marker_map(self, marker_map: np.ndarray) -> None:
        if marker_map.shape != self.markers.shape:
            raise ValueError("marker map shape mismatch")
        self.markers = marker_map.astype(np.int8, copy=True)

    # ---------------------------------------------------------
    # FIXED MAZE: fully connected, no sealed chambers
    # ---------------------------------------------------------
    def _make_grid(self, walls_layout: str) -> np.ndarray:
        if walls_layout != "fixed_maze_1":
            raise ValueError("only walls_layout='fixed_maze_1' is implemented")

        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Outer walls
        grid[0, :] = self.WALL
        grid[-1, :] = self.WALL
        grid[:, 0] = self.WALL
        grid[:, -1] = self.WALL

        # Simple connected maze: a single vertical wall with two openings
        if self.grid_size >= 8:
            mid = self.grid_size // 2

            # Vertical wall
            grid[2:self.grid_size - 2, mid] = self.WALL

            # Two openings to ensure connectivity
            grid[3, mid] = self.EMPTY
            grid[self.grid_size - 4, mid] = self.EMPTY

        return grid

    def _empty_cells(self) -> List[Position]:
        rows, cols = np.where(self.grid == self.EMPTY)
        return list(zip(rows.tolist(), cols.tolist()))

    def _is_wall_or_oob(self, row: int, col: int) -> bool:
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return True
        return bool(self.grid[row, col] == self.WALL)

    def _get_obs(self) -> np.ndarray:
        half = self.view_size // 2
        channels = 4 if self.has_markers else 3
        obs = np.zeros((self.view_size, self.view_size, channels), dtype=np.float32)

        for vr in range(self.view_size):
            for vc in range(self.view_size):
                gr = self.agent_pos[0] + vr - half
                gc = self.agent_pos[1] + vc - half

                if self._is_wall_or_oob(gr, gc):
                    obs[vr, vc, 0] = 1.0
                    continue

                if (gr, gc) == self.goal_pos:
                    obs[vr, vc, 1] = 1.0
                if (gr, gc) == self.agent_pos:
                    obs[vr, vc, 2] = 1.0
                if self.has_markers and self.markers[gr, gc] == 1:
                    obs[vr, vc, 3] = 1.0

        return obs
