from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ---------------------------------------------------------
# Utility: convert numpy obs to torch tensor
# ---------------------------------------------------------
def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if x.ndim == 3:
        # (H, W, C) -> (1, C, H, W)
        x = x.permute(2, 0, 1).unsqueeze(0)
    elif x.ndim == 4:
        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
    return x


# ---------------------------------------------------------
# Policy + Value Network
# ---------------------------------------------------------
class PolicyValueNet(nn.Module):
    def __init__(self, input_channels: int, view_size: int, num_actions: int) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # After two 3x3 convs with stride 1, spatial dims shrink by 2 each time
        conv_out = 64 * (view_size - 4) * (view_size - 4)

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 128),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.fc(self.trunk(x))
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value


# ---------------------------------------------------------
# PPO Config
# ---------------------------------------------------------
@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    batch_size: int = 64
    rollout_steps: int = 2048
    ppo_epochs: int = 10


# ---------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------
class PPOAgent:
    def __init__(
        self,
        input_channels: int,
        view_size: int,
        num_actions: int,
        config: Dict[str, float] | PPOConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:

        # Load config
        if isinstance(config, PPOConfig):
            self.config = config
        else:
            self.config = PPOConfig(**(config or {}))

        # -------------------------------------------------
        # FIX: enforce numeric types (YAML sometimes returns strings)
        # -------------------------------------------------
        self.config.lr = float(self.config.lr)
        self.config.gamma = float(self.config.gamma)
        self.config.gae_lambda = float(self.config.gae_lambda)
        self.config.clip_range = float(self.config.clip_range)
        self.config.entropy_coef = float(self.config.entropy_coef)
        self.config.value_coef = float(self.config.value_coef)
        self.config.batch_size = int(self.config.batch_size)
        self.config.rollout_steps = int(self.config.rollout_steps)
        self.config.ppo_epochs = int(self.config.ppo_epochs)

        # Device
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Model + Optimizer
        self.model = PolicyValueNet(input_channels, view_size, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

    # -----------------------------------------------------
    # Action selection
    # -----------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        x = obs_to_tensor(obs, self.device)
        logits, value = self.model(x)
        dist = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    # -----------------------------------------------------
    # PPO Update
    # -----------------------------------------------------
    def update(self, rollout: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = obs_to_tensor(rollout["obs"], self.device)
        actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(rollout["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        n = actions.shape[0]
        losses, policy_losses, value_losses, entropies = [], [], [], []

        for _ in range(self.config.ppo_epochs):
            indices = torch.randperm(n, device=self.device)

            for start in range(0, n, self.config.batch_size):
                mb_idx = indices[start : start + self.config.batch_size]

                logits, values = self.model(obs[mb_idx])
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(actions[mb_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - old_log_probs[mb_idx])
                unclipped = ratio * advantages[mb_idx]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                ) * advantages[mb_idx]

                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (returns[mb_idx] - values).pow(2).mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                losses.append(float(loss.item()))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        return {
            "loss": float(np.mean(losses)),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }

    # -----------------------------------------------------
    # Save / Load
    # -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path: str | Path, load_optimizer: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


# ---------------------------------------------------------
# GAE Computation
# ---------------------------------------------------------
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(len(rewards))):
        next_nonterminal = 1.0 - dones[t]
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------
# Rollout Collection
# ---------------------------------------------------------
def collect_rollout(env, policy: PPOAgent, rollout_steps: int) -> Dict[str, np.ndarray | List[Dict[str, object]]]:
    obs_buf, action_buf, log_prob_buf = [], [], []
    reward_buf, done_buf, value_buf = [], [], []
    episode_logs: List[Dict[str, object]] = []

    obs = getattr(env, "_current_obs", None)
    if obs is None:
        obs = env.reset()

    episode_return = float(getattr(env, "_episode_return", 0.0))
    episode_index = int(getattr(env, "_episode_index", 0))

    for _ in range(rollout_steps):
        action, log_prob, value = policy.act(obs)
        next_obs, reward, done, info = env.step(action)

        obs_buf.append(obs)
        action_buf.append(action)
        log_prob_buf.append(log_prob)
        reward_buf.append(reward)
        done_buf.append(float(done))
        value_buf.append(value)

        episode_return += float(reward)
        obs = next_obs

        if done:
            episode_logs.append(
                {
                    "episode_index": episode_index,
                    "total_return": episode_return,
                    "success": int(info.get("success", 0)),
                    "steps": int(info.get("steps", 0)),
                    "markers_used": int(info.get("markers_used", 0)),
                }
            )
            episode_index += 1
            episode_return = 0.0
            obs = env.reset()

    with torch.no_grad():
        _, last_value_tensor = policy.model(obs_to_tensor(obs, policy.device))
    last_value = float(last_value_tensor.item())

    advantages, returns = compute_gae(
        np.asarray(reward_buf, dtype=np.float32),
        np.asarray(value_buf, dtype=np.float32),
        np.asarray(done_buf, dtype=np.float32),
        last_value,
        policy.config.gamma,
        policy.config.gae_lambda,
    )

    env._current_obs = obs
    env._episode_return = episode_return
    env._episode_index = episode_index

    return {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "actions": np.asarray(action_buf, dtype=np.int64),
        "log_probs": np.asarray(log_prob_buf, dtype=np.float32),
        "rewards": np.asarray(reward_buf, dtype=np.float32),
        "dones": np.asarray(done_buf, dtype=np.float32),
        "values": np.asarray(value_buf, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "episode_logs": episode_logs,
    }
