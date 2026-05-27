from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

from mate.agents.ppo import PPOAgent
from mate.envs.gridworld import GridWorldEnv
from mate.logging.logger import append_row_csv
from mate.logging.utils import ensure_dir, make_run_id, set_global_seeds
from mate.training.train import load_yaml, make_env


ROLLOUT_FIELDS = [
    "run_id",
    "condition",
    "seed",
    "intervention",
    "episode_index",
    "t",
    "action",
    "reward",
    "done",
    "agent_x",
    "agent_y",
    "goal_x",
    "goal_y",
    "marker_map",
    "state_hash",
]

MARKER_FIELDS = [
    "run_id",
    "condition",
    "seed",
    "intervention",
    "episode_index",
    "marker_map",
    "placements",
]


def zero_marker_channel(obs: np.ndarray) -> np.ndarray:
    obs = obs.copy()
    if obs.shape[-1] == 4:
        obs[:, :, 3] = 0.0
    return obs


def shuffle_markers(env: GridWorldEnv) -> None:
    flat = env.marker_map().reshape(-1).copy()
    env.rng.shuffle(flat)
    env.set_marker_map(flat.reshape(env.grid_size, env.grid_size))


def observation_hash(obs: np.ndarray) -> str:
    return hashlib.sha1(obs.astype(np.float32).tobytes()).hexdigest()


def serialize_array(array: np.ndarray) -> str:
    return json.dumps(array.astype(int).tolist())


def apply_intervention_obs(env: GridWorldEnv, obs: np.ndarray, intervention_type: str) -> np.ndarray:
    if not env.has_markers:
        return obs
    if intervention_type == "marker_deletion":
        return zero_marker_channel(obs)
    return obs


def eval_seed(
    env_config_path: Path,
    agent_config_path: Path,
    eval_config: Dict,
    condition: str,
    checkpoint_dir: Path,
    output_dir: Path,
    seed: int,
) -> None:
    env_config = load_yaml(env_config_path)
    agent_config = load_yaml(agent_config_path)
    run_id = make_run_id(condition, seed)
    checkpoint_path = checkpoint_dir / run_id / "final.pt"
    if not checkpoint_path.exists():
        return

    set_global_seeds(seed)
    env = make_env(env_config, condition, seed)
    agent = PPOAgent(
        input_channels=env.observation_shape[-1],
        view_size=env.view_size,
        num_actions=env.num_actions,
        config=agent_config["ppo"],
    )
    agent.load(checkpoint_path, load_optimizer=False)

    processed_dir = output_dir.parent / "processed"
    rollouts_path = processed_dir / "rollouts.csv"
    markers_path = processed_dir / "markers.csv"
    num_eval_episodes = int(eval_config["num_eval_episodes"])

    for intervention in eval_config["interventions"]:
        intervention_name = intervention["name"]
        intervention_type = intervention["type"]
        for episode_index in range(num_eval_episodes):
            obs = env.reset()
            if env.has_markers and intervention_type == "marker_shuffle":
                shuffle_markers(env)
                obs = env._get_obs()
            obs = apply_intervention_obs(env, obs, intervention_type)

            done = False
            t = 0
            while not done:
                if env.has_markers and intervention_type == "marker_shuffle":
                    shuffle_markers(env)
                    obs = env._get_obs()
                    obs = apply_intervention_obs(env, obs, intervention_type)

                state_hash = observation_hash(obs)
                action, _, _ = agent.act(obs, deterministic=True)
                next_obs, reward, done, info = env.step(action)
                marker_map = env.marker_map()
                agent_x, agent_y = info["agent_pos"]
                goal_x, goal_y = info["goal_pos"]

                append_row_csv(
                    rollouts_path,
                    {
                        "run_id": run_id,
                        "condition": condition,
                        "seed": seed,
                        "intervention": intervention_name,
                        "episode_index": episode_index,
                        "t": t,
                        "action": action,
                        "reward": reward,
                        "done": int(done),
                        "agent_x": agent_x,
                        "agent_y": agent_y,
                        "goal_x": goal_x,
                        "goal_y": goal_y,
                        "marker_map": serialize_array(marker_map),
                        "state_hash": state_hash,
                    },
                    ROLLOUT_FIELDS,
                )

                obs = apply_intervention_obs(env, next_obs, intervention_type)
                t += 1

            append_row_csv(
                markers_path,
                {
                    "run_id": run_id,
                    "condition": condition,
                    "seed": seed,
                    "intervention": intervention_name,
                    "episode_index": episode_index,
                    "marker_map": serialize_array(env.marker_map()),
                    "placements": json.dumps(info.get("placements", [])),
                },
                MARKER_FIELDS,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoints on MATE GridWorld.")
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--condition", choices=["C0", "C1"], required=True)
    parser.add_argument("--checkpoint-dir", default="runs/raw/")
    parser.add_argument("--output-dir", default="runs/raw/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_config = load_yaml(args.eval_config)
    agent_config = load_yaml(args.agent_config)
    output_dir = ensure_dir(args.output_dir)
    for seed in agent_config["seeds"]:
        eval_seed(
            Path(args.env_config),
            Path(args.agent_config),
            eval_config,
            args.condition,
            Path(args.checkpoint_dir),
            Path(output_dir),
            int(seed),
        )


if __name__ == "__main__":
    main()
