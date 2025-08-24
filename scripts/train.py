import argparse
import os
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

from env.kf_env import KFEnv
from env.entities.agent import Agent
from env.entities.stable_obstacle import StableObstacle
from extractors.attention_extractor import AttentionExtractor
from sac import KFSACPolicy


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAC Training with Attention Extractor"
    )

    # Environment
    parser.add_argument("--max-obstacles", type=int, default=10)
    parser.add_argument("--target-radius", type=float, default=10.0)
    parser.add_argument("--recognition-radius", type=float, default=100.0)
    parser.add_argument("--destruction-radius", type=float, default=200.0)
    parser.add_argument("--n-obstacles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Feature Extractor
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--include-acceleration", action="store_true")
    parser.add_argument("--features-dim", type=int, default=128)

    # SAC
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--ent-coef", default="auto")
    parser.add_argument("--log-std-init", type=float, default=-3.0)

    # Network
    parser.add_argument("--net-arch", nargs="+", type=int, default=[256, 256])

    # Training
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-path", default="sac_attention")
    parser.add_argument("--log-interval", type=int, default=1000)

    # Logging & Video Recording
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="./logs/",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50_000,
        help="Model checkpoint save frequency",
    )
    parser.add_argument(
        "--record-video", action="store_true", help="Record training videos"
    )
    parser.add_argument(
        "--video-freq",
        type=int,
        default=50_000,
        help="Video recording frequency",
    )
    parser.add_argument(
        "--video-length", type=int, default=1000, help="Video length in steps"
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default=None,
        choices=[None, "human", "rgb_array"],
        help="Render mode for environment",
    )

    # Device
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--verbose", type=int, default=1)

    return parser.parse_args()


def make_env(args, eval_mode=False):
    """환경 생성 함수"""

    def _init():
        render_mode = (
            "rgb_array" if eval_mode and args.record_video else args.render_mode
        )
        env = KFEnv(
            max_obstacles=args.max_obstacles,
            target_radius=args.target_radius,
            recognition_radius=args.recognition_radius,
            destruction_radius=args.destruction_radius,
            render_mode=render_mode,
        )

        env.add_agent(agent_class=Agent)

        for _ in range(args.n_obstacles):
            env.add_obstacle(obstacle_class=StableObstacle)

        return env

    return _init


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    # 로그 디렉토리 생성
    os.makedirs(args.tensorboard_log, exist_ok=True)

    # save_path 디렉토리 생성 (파일명에 디렉토리가 포함된 경우만)
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("Creating environment...")
    env = KFEnv(
        max_obstacles=args.max_obstacles,
        target_radius=args.target_radius,
        recognition_radius=args.recognition_radius,
        destruction_radius=args.destruction_radius,
        render_mode=args.render_mode,
    )

    print("Adding agent...")
    env.add_agent(agent_class=Agent)

    print(f"Adding {args.n_obstacles} obstacles...")
    for _ in range(args.n_obstacles):
        env.add_obstacle(obstacle_class=StableObstacle)

    print("Testing environment reset...")
    obs, info = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(
        f"Reset observation keys: {obs.keys() if isinstance(obs, dict) else type(obs)}"
    )
    print("Environment test successful!")

    print("Creating policy kwargs...")
    policy_kwargs = {
        "net_arch": args.net_arch,
        "activation_fn": nn.ReLU,
        "log_std_init": args.log_std_init,
        "features_extractor_class": AttentionExtractor,
        "features_extractor_kwargs": {
            "max_obstacles": args.max_obstacles,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "include_acceleration": args.include_acceleration,
            "features_dim": args.features_dim,
        },
    }

    print("Creating SAC model...")
    model = SAC(
        policy=KFSACPolicy,
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        learning_starts=args.learning_starts,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        device=args.device,
        verbose=args.verbose,
        tensorboard_log=args.tensorboard_log,
    )

    print(f"Training SAC with AttentionExtractor")
    print(
        f"Environment: {args.max_obstacles} max obstacles, {args.n_obstacles} spawned, "
        f"target_radius={args.target_radius}, recognition_radius={args.recognition_radius}, "
        f"destruction_radius={args.destruction_radius}"
    )
    print(
        f"Extractor: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}"
    )
    print(f"Network: {args.net_arch}, activation=ReLU")
    print(f"Total timesteps: {args.total_timesteps}")

    print("Testing model with a few steps...")
    try:
        obs, info = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"Environment step test successful! Reward: {reward}")

        callbacks = []

        checkpoint_callback = CheckpointCallback(
            save_freq=args.save_freq,
            save_path=f"{args.save_path}_checkpoints/",
            name_prefix="sac_checkpoint",
        )
        callbacks.append(checkpoint_callback)

        if args.eval_freq > 0:
            eval_env = make_vec_env(make_env(args, eval_mode=True), n_envs=1)

            if args.record_video:
                try:
                    eval_env = VecVideoRecorder(
                        eval_env,
                        f"{args.tensorboard_log}/videos/",
                        record_video_trigger=lambda x: x % args.video_freq == 0,
                        video_length=args.video_length,
                    )
                    print("Video recording enabled!")
                except Exception as e:
                    print(f"Video recording failed: {e}")
                    print("Continuing without video recording...")

            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=f"{args.save_path}_best/",
                log_path=f"{args.tensorboard_log}/evaluations/",
                eval_freq=args.eval_freq,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)

        print("Starting learning...")
        model.learn(
            total_timesteps=args.total_timesteps,
            log_interval=args.log_interval,
            callback=callbacks if callbacks else None,
            tb_log_name="SAC_AttentionExtractor",
        )
    except Exception as e:
        print(f"Error during learning: {e}")
        import traceback

        traceback.print_exc()
        raise

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
