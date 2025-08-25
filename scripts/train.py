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
from scripts.configs import (
    get_config,
    list_available_presets,
)

# =============================================================================
# Configuration Selection
# =============================================================================

# Choose your configuration preset here:
CONFIG_PRESET = "standard"  # Change this to use different configurations

config = get_config(CONFIG_PRESET)


def make_env(is_eval=False):
    def _init():
        # Determine render mode based on video recording settings
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
        },
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
