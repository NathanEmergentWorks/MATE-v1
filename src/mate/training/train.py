from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import yaml

from mate.agents.ppo import PPOAgent, collect_rollout
from mate.envs.gridworld import GridWorldEnv
from mate.logging.logger import append_row_csv
from mate.logging.utils import ensure_dir, make_run_id, set_global_seeds


EPISODE_FIELDS = [
    "run_id",
    "condition",
    "seed",
    "episode_index",
    "total_return",
    "success",
    "steps",
    "markers_used",
    "env_config_path",
    "agent_config_path",
]

TRAIN_METRIC_FIELDS = [
    "run_id",
    "condition",
    "seed",
    "env_steps",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
]


def load_yaml(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(env_config: Dict, condition: str, seed: int) -> GridWorldEnv:
    cfg = dict(env_config)
    cfg.pop("name", None)
    cfg["has_markers"] = condition == "C1"
    cfg["marker_persistence"] = "episode" if condition == "C1" else "none"
    env = GridWorldEnv(**cfg)
    env.seed(seed)
    return env


def train_seed(
    env_config_path: Path,
    agent_config_path: Path,
    condition: str,
    output_dir: Path,
    seed: int,
) -> None:
    env_config = load_yaml(env_config_path)
    agent_config = load_yaml(agent_config_path)
    run_id = make_run_id(condition, seed)

    set_global_seeds(seed)
    env = make_env(env_config, condition, seed)
    ppo_cfg = agent_config["ppo"]
    agent = PPOAgent(
        input_channels=env.observation_shape[-1],
        view_size=env.view_size,
        num_actions=env.num_actions,
        config=ppo_cfg,
    )

    total_env_steps = int(agent_config["training"]["total_env_steps"])
    eval_interval_steps = int(agent_config["training"].get("eval_interval_steps", total_env_steps))
    rollout_steps = int(ppo_cfg["rollout_steps"])

    run_dir = output_dir / run_id
    ensure_dir(run_dir)
    processed_dir = output_dir.parent / "processed"
    episodes_path = processed_dir / "episodes.csv"
    train_metrics_path = processed_dir / "train_metrics.csv"

    env_steps = 0
    next_checkpoint = eval_interval_steps

    while env_steps < total_env_steps:
        this_rollout_steps = min(rollout_steps, total_env_steps - env_steps)
        rollout = collect_rollout(env, agent, this_rollout_steps)
        metrics = agent.update(rollout)
        env_steps += this_rollout_steps

        append_row_csv(
            train_metrics_path,
            {
                "run_id": run_id,
                "condition": condition,
                "seed": seed,
                "env_steps": env_steps,
                **metrics,
            },
            TRAIN_METRIC_FIELDS,
        )

        for ep in rollout["episode_logs"]:
            append_row_csv(
                episodes_path,
                {
                    "run_id": run_id,
                    "condition": condition,
                    "seed": seed,
                    "episode_index": ep["episode_index"],
                    "total_return": ep["total_return"],
                    "success": ep["success"],
                    "steps": ep["steps"],
                    "markers_used": ep["markers_used"] if condition == "C1" else 0,
                    "env_config_path": str(env_config_path),
                    "agent_config_path": str(agent_config_path),
                },
                EPISODE_FIELDS,
            )

        if env_steps >= next_checkpoint:
            agent.save(run_dir / f"checkpoint_{env_steps}.pt")
            next_checkpoint += eval_interval_steps

    agent.save(run_dir / "final.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on MATE GridWorld.")
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--condition", choices=["C0", "C1"], required=True)
    parser.add_argument("--output-dir", default="runs/raw/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    agent_config = load_yaml(args.agent_config)
    for seed in agent_config["seeds"]:
        train_seed(
            Path(args.env_config),
            Path(args.agent_config),
            args.condition,
            Path(output_dir),
            int(seed),
        )


if __name__ == "__main__":
    main()
