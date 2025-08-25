import time
import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

from sac import KFSACPolicy
from env.kf_env import KFEnv
from env.entities.agent import Agent
from env.entities.stable_obstacle import StableObstacle
from extractors.attention_extractor import AttentionExtractor
from extractors.lstm_extractor import LSTMExtractor
from extractors.padding_extractor import PaddingExtractor
from scripts.configs import (
    get_config,
    list_available_presets,
)

# =============================================================================
# Configuration Selection
# =============================================================================

# Choose your configuration preset here:
CONFIG_PRESET = "attention"  # Change this to use different configurations

config = get_config(CONFIG_PRESET)


def get_extractor_config(feature_extractor_config, environment_config):
    """Get the appropriate feature extractor class and kwargs based on configuration."""

    # Common kwargs for all extractors
    common_kwargs = {
        "max_obstacles": environment_config.max_obstacles,
        "include_acceleration": feature_extractor_config.include_acceleration,
        "include_radius": feature_extractor_config.include_radius,
    }

    if feature_extractor_config.extractor_type == "attention":
        return AttentionExtractor, {
            **common_kwargs,
            "d_model": feature_extractor_config.d_model,
            "n_heads": feature_extractor_config.n_heads,
            "n_layers": feature_extractor_config.n_layers,
            "dropout": feature_extractor_config.dropout,
        }
    elif feature_extractor_config.extractor_type == "lstm":
        return LSTMExtractor, {
            **common_kwargs,
            "lstm_hidden": feature_extractor_config.lstm_hidden,
            "lstm_layers": feature_extractor_config.lstm_layers,
            "bidirectional": feature_extractor_config.bidirectional,
            "use_layernorm": feature_extractor_config.use_layernorm,
            "features_dim": feature_extractor_config.features_dim,
        }
    elif feature_extractor_config.extractor_type == "padding":
        return PaddingExtractor, common_kwargs
    else:
        raise ValueError(
            f"Unknown extractor type: {feature_extractor_config.extractor_type}"
        )


def make_env(is_eval=False):
    def _init():
        if is_eval and config.eval_video.record_video:
            render_mode = "rgb_array"
        elif not is_eval and config.training_video.record_video:
            render_mode = "rgb_array"
        else:
            render_mode = config.environment.render_mode

        env = KFEnv(
            max_obstacles=config.environment.max_obstacles,
            target_radius=config.environment.target_radius,
            recognition_radius=config.environment.recognition_radius,
            destruction_radius=config.environment.destruction_radius,
            max_velocity=config.environment.max_velocity,
            max_acceleration=config.environment.max_acceleration,
            render_mode=render_mode,
        )

        env.add_agent(agent_class=Agent)

        for _ in range(config.environment.n_obstacles):
            env.add_obstacle(obstacle_class=StableObstacle)

        return env

    return _init


def main():
    print("Setting random seed...")
    torch.manual_seed(config.training.seed)

    print("Creating directories...")
    directories_to_create = [
        config.logging.logs_dir,
        config.logging.tensorboard_dir,
        config.checkpoint.checkpoint_dir,
        config.eval.log_dir,
    ]

    if config.training_video.record_training:
        directories_to_create.append(config.training_video.trains_dir)

    if config.eval_video.record_eval:
        directories_to_create.extend(
            [
                config.eval_video.evals_dir,
                (
                    config.eval_video.best_dir
                    if config.eval_video.record_best
                    else None
                ),
            ]
        )

    directories_to_create = [d for d in directories_to_create if d is not None]
    for directory in directories_to_create:
        os.makedirs(directory, exist_ok=True)
    print(f"Created {len(directories_to_create)} directories")

    print("Creating training environments...")
    env = make_vec_env(
        make_env(is_eval=False),
        n_envs=config.training.n_envs,
        seed=config.training.seed,
    )
    print(f"Created {config.training.n_envs} parallel environments")

    if config.training_video.record_video:
        print("Setting up training video recording...")
        env = VecVideoRecorder(
            env,
            config.training_video.trains_dir,
            record_video_trigger=lambda x: x % config.training_video.video_freq
            == 0,
            video_length=config.training_video.video_length,
            name_prefix="train",
        )
        print("Training video recording enabled")

    print("Creating policy configuration...")
    extractor_class, extractor_kwargs = get_extractor_config(
        config.feature_extractor, config.environment
    )

    policy_kwargs = {
        "net_arch": config.network.net_arch,
        "activation_fn": config.network.activation_fn,
        "log_std_init": config.sac.log_std_init,
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": extractor_kwargs,
    }

    print("Creating SAC model...")
    model = SAC(
        policy=KFSACPolicy,
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
        tensorboard_log=config.logging.tensorboard_dir,
    )
    print("SAC model created successfully")

    try:
        print("Setting up callbacks...")
        callbacks = []

        checkpoint_callback = CheckpointCallback(
            save_freq=config.checkpoint.save_freq,
            save_path=config.checkpoint.checkpoint_dir,
            name_prefix="cp",
        )
        callbacks.append(checkpoint_callback)
        print("Checkpoint callback added")

        if config.eval.eval_freq > 0:
            print("Creating evaluation environment...")
            eval_env = make_vec_env(
                make_env(is_eval=True), n_envs=config.eval.n_eval_envs
            )

            if config.eval_video.record_video:
                print("Setting up evaluation video recording...")
                eval_env = VecVideoRecorder(
                    eval_env,
                    config.eval_video.evals_dir,
                    record_video_trigger=lambda x: x
                    % config.eval_video.video_freq
                    == 0,
                    video_length=config.eval_video.video_length,
                    name_prefix="eval",
                )

            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.dirname(
                    config.eval.best_model_path
                ),
                log_path=config.eval.log_dir,
                eval_freq=config.eval.eval_freq,
                n_eval_episodes=config.eval.eval_episodes,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)
            print("Evaluation callback added")

        print("Starting training...")
        model.learn(
            total_timesteps=config.training.total_timesteps,
            log_interval=config.training.log_interval,
            callback=callbacks if callbacks else None,
            tb_log_name=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
        )
        print("Training completed successfully")

    except Exception as e:
        print(f"Error during learning: {e}")
        import traceback

        traceback.print_exc()
        raise

    print("Saving final model...")
    model.save(config.checkpoint.latest_model_path)
    print(f"Model saved to: {config.checkpoint.latest_model_path}")

    print("Closing environments...")
    env.close()
    print("Training finished")


if __name__ == "__main__":
    print("Available configuration presets:")
    for preset in list_available_presets():
        print(f"  - {preset}")
    print(f"\nCurrently using: '{CONFIG_PRESET}'\n")

    main()
