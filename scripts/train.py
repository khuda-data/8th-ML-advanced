"""
Main training script for KHUDAFinder using configuration-based setup.
Uses predefined configuration presets for different training scenarios.
"""

import time
import os
import torch
import warnings
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

# Suppress specific warnings about environment type mismatch
warnings.filterwarnings(
    "ignore", message=".*Training and eval env are not of the same type.*"
)

from env.kf_env import KFEnv
from env.entities.agent import Agent
from env.entities.stable_obstacle import StableObstacle
from extractors.attention_extractor import AttentionExtractor
from scripts.configs import (
    get_config,
    print_config_summary,
    list_available_presets,
)

# =============================================================================
# Configuration Selection
# =============================================================================

# Choose your configuration preset here:
CONFIG_PRESET = "quick_test"  # Change this to use different configurations

config = get_config(CONFIG_PRESET)


def make_env(eval_mode=False):
    """
    Create environment factory function for vectorized environments.

    Args:
        eval_mode (bool): If True, creates environment for evaluation with video recording

    Returns:
        callable: Environment factory function that returns initialized KFEnv
    """

    def _init():
        render_mode = (
            "rgb_array"
            if eval_mode and config.video.record_video
            else config.environment.render_mode
        )
        env = KFEnv(
            max_obstacles=config.environment.max_obstacles,
            target_radius=config.environment.target_radius,
            recognition_radius=config.environment.recognition_radius,
            destruction_radius=config.environment.destruction_radius,
            render_mode=render_mode,
        )

        env.add_agent(agent_class=Agent)

        for _ in range(config.environment.n_obstacles):
            env.add_obstacle(obstacle_class=StableObstacle)

        return env

    return _init


def main():
    print(f"Using configuration preset: '{CONFIG_PRESET}'")
    print_config_summary(config)

    torch.manual_seed(config.training.seed)

    os.makedirs(config.logging.tensorboard_log, exist_ok=True)

    save_dir = os.path.dirname(config.logging.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("Creating vectorized training environment...")
    env = make_vec_env(
        make_env(eval_mode=False),
        n_envs=config.training.n_envs,
        seed=config.training.seed,
    )

    print(
        f"Created {config.training.n_envs} parallel environments for training"
    )

    print("Creating policy kwargs...")
    policy_kwargs = {
        "net_arch": config.network.net_arch,
        "activation_fn": config.network.activation_fn,
        "log_std_init": config.sac.log_std_init,
        "features_extractor_class": AttentionExtractor,
        "features_extractor_kwargs": {
            "max_obstacles": config.environment.max_obstacles,
            "d_model": config.feature_extractor.d_model,
            "n_heads": config.feature_extractor.n_heads,
            "n_layers": config.feature_extractor.n_layers,
            "dropout": config.feature_extractor.dropout,
            "include_acceleration": config.feature_extractor.include_acceleration,
            "features_dim": config.feature_extractor.features_dim,
        },
    }

    print("Creating SAC model...")
    model = SAC(
        policy="MultiInputPolicy",  # Policy for Dict observation spaces
        env=env,
        learning_rate=config.sac.learning_rate,
        buffer_size=config.sac.buffer_size,
        batch_size=config.sac.batch_size,
        tau=config.sac.tau,
        gamma=config.sac.gamma,
        learning_starts=config.sac.learning_starts,
        ent_coef=config.sac.ent_coef,
        policy_kwargs=policy_kwargs,
        device=config.device.device,
        verbose=config.logging.verbose,
        tensorboard_log=config.logging.tensorboard_log,
    )

    print("Testing vectorized model with sample actions...")
    try:
        callbacks = []

        checkpoint_callback = CheckpointCallback(
            save_freq=config.logging.save_freq,
            save_path=f"{config.logging.save_path}/checkpoints/",
            name_prefix="checkpoint",
        )
        callbacks.append(checkpoint_callback)

        if config.training.eval_freq > 0:
            eval_env = make_vec_env(
                make_env(eval_mode=True), n_envs=config.training.n_eval_envs
            )

            if config.video.record_video:
                video_dir = f"{config.logging.tensorboard_log}/evals/"
                os.makedirs(video_dir, exist_ok=True)

                eval_env = VecVideoRecorder(
                    eval_env,
                    video_dir,
                    record_video_trigger=lambda x: x % config.video.video_freq
                    == 0,
                    video_length=config.video.video_length,
                )

            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=f"{config.logging.save_path}/best/",
                log_path=f"{config.logging.tensorboard_log}/evals/",
                eval_freq=config.training.eval_freq,
                n_eval_episodes=config.training.eval_episodes,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)

        print("Starting learning...")
        model.learn(
            total_timesteps=config.training.total_timesteps,
            log_interval=config.training.log_interval,
            callback=callbacks if callbacks else None,
            tb_log_name=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
        )

    except Exception as e:
        print(f"Error during learning: {e}")
        import traceback

        traceback.print_exc()
        raise

    model.save(config.logging.save_path)

    env.close()


if __name__ == "__main__":
    # Print available presets for user reference
    print("Available configuration presets:")
    for preset in list_available_presets():
        print(f"  - {preset}")
    print(f"\nCurrently using: '{CONFIG_PRESET}'\n")

    main()
