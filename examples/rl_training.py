"""Reinforcement learning training — reference implementation.

Uses Stable Baselines 3. Demonstrates the project code quality contract:
- Complete type hints on all signatures
- Typed config via dataclass
- argparse CLI entry point
- logging (no print)
- Modular structure: load_data(env) / build_model / train / evaluate / export
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    env_id: str = "LunarLander-v2"
    output_dir: Path = Path("outputs/rl")
    total_timesteps: int = 1_000_000
    n_envs: int = 8
    eval_freq: int = 10_000
    n_eval_episodes: int = 20
    reward_threshold: float = 200.0
    seed: int = 42
    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy: str = "MlpPolicy"


def make_env(cfg: Config) -> tuple[VecEnv, gym.Env]:
    """Create vectorised training env and single eval env."""
    train_env = make_vec_env(cfg.env_id, n_envs=cfg.n_envs, seed=cfg.seed)
    eval_env = Monitor(gym.make(cfg.env_id))
    logger.info(
        "Env: %s | n_envs: %d | obs_space: %s | action_space: %s",
        cfg.env_id, cfg.n_envs,
        train_env.observation_space, train_env.action_space,
    )
    return train_env, eval_env


def build_model(train_env: VecEnv, cfg: Config) -> PPO:
    """Instantiate PPO with project config."""
    model = PPO(
        policy=cfg.policy,
        env=train_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        seed=cfg.seed,
        verbose=1,
    )
    logger.info("PPO model built | policy: %s | device: %s", cfg.policy, model.device)
    return model


def train(model: PPO, eval_env: gym.Env, cfg: Config) -> PPO:
    """Train with early stopping on reward threshold."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = str(cfg.output_dir / "best_model")

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=cfg.reward_threshold, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=str(cfg.output_dir / "eval_logs"),
        eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        callback_on_new_best=stop_callback,
    )

    model.learn(total_timesteps=cfg.total_timesteps, callback=eval_callback)
    logger.info("Training complete | timesteps: %d", model.num_timesteps)
    return model


def evaluate(model: PPO, eval_env: gym.Env, cfg: Config) -> dict[str, float]:
    """Evaluate final policy and log mean/std reward."""
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=cfg.n_eval_episodes, deterministic=True
    )
    metrics = {"mean_reward": float(mean_reward), "std_reward": float(std_reward)}
    logger.info("Evaluation | mean_reward=%.2f ± %.2f", mean_reward, std_reward)
    threshold_met = mean_reward >= cfg.reward_threshold
    logger.info("Reward threshold %.1f %s", cfg.reward_threshold,
                "MET ✓" if threshold_met else "NOT MET ✗")
    return metrics


def export(model: PPO, cfg: Config) -> Path:
    """Save final model to output_dir/final_model.zip."""
    export_path = cfg.output_dir / "final_model"
    model.save(str(export_path))
    logger.info("Model saved to %s.zip", export_path)
    return Path(str(export_path) + ".zip")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="RL training with Stable Baselines 3 PPO")
    parser.add_argument("--env-id", type=str, default="LunarLander-v2")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rl"))
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--reward-threshold", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()
    return Config(
        env_id=args.env_id,
        output_dir=args.output_dir,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        reward_threshold=args.reward_threshold,
        seed=args.seed,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
    )


def main() -> None:
    cfg = parse_args()
    logger.info("Config: %s", cfg)
    train_env, eval_env = make_env(cfg)
    model = build_model(train_env, cfg)
    train(model, eval_env, cfg)
    evaluate(model, eval_env, cfg)
    export(model, cfg)
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
