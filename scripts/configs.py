"""
Configuration classes and predefined settings for KHUDAFinder training.
Provides organized configuration management with different presets for various scenarios.
"""

from dataclasses import dataclass
from typing import Optional, List, Union
import torch.nn as nn


@dataclass
class EnvironmentConfig:
    """Environment-related configuration settings."""

    max_obstacles: int = 20
    target_radius: float = 1.0
    recognition_radius: float = 10.0
    destruction_radius: float = 25.0
    n_obstacles: int = 20
    render_mode: Optional[str] = None  # None, "human", "rgb_array"


@dataclass
class FeatureExtractorConfig:
    """Feature extractor (attention mechanism) configuration."""

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    include_acceleration: bool = True


@dataclass
class SACConfig:
    """SAC algorithm hyperparameters configuration."""

    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    learning_starts: int = 5_000
    ent_coef: Union[str, float] = "auto"
    log_std_init: float = -3.0


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""

    net_arch: List[int] = None
    activation_fn = nn.ReLU

    def __post_init__(self):
        if self.net_arch is None:
            self.net_arch = [256, 256]


@dataclass
class TrainingConfig:
    """Training process configuration."""

    total_timesteps: int = 1_000_000
    log_interval: int = 1_000
    n_envs: int = 8
    seed: int = 42


@dataclass
class EvalConfig:
    """Evaluation configuration for model assessment."""

    eval_freq: int = 10_000
    eval_episodes: int = 5
    n_eval_envs: int = 2

    eval_dir: str = "evals"
    best_model_path: str = None
    log_dir: str = None
    metric_path: str = None
    plots_dir: str = None

    def __post_init__(self):
        if self.best_model_path is None:
            self.best_model_path = f"{self.eval_dir}/best.zip"
        if self.log_dir is None:
            self.log_dir = f"{self.eval_dir}/logs"
        if self.metric_path is None:
            self.metric_path = f"{self.eval_dir}/metrics.json"
        if self.plots_dir is None:
            self.plots_dir = f"{self.eval_dir}/plots"


@dataclass
class CheckPointConfig:
    """Checkpoint saving and loading configuration."""

    save_freq: int = 10_000
    max_checkpoints: int = 5

    checkpoint_dir: str = "checkpoints"
    latest_model_path: str = None
    backup_dir: str = None

    def __post_init__(self):
        if self.latest_model_path is None:
            self.latest_model_path = f"{self.checkpoint_dir}/latest.zip"
        if self.backup_dir is None:
            self.backup_dir = f"{self.checkpoint_dir}/backups"


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""

    verbose: int = 1

    logs_dir: str = "logs"
    trains_path: str = None
    errors_path: str = None
    tensorboard_dir: str = None

    def __post_init__(self):
        if self.tensorboard_dir is None:
            self.tensorboard_dir = f"{self.logs_dir}/tensorboard/"
        if self.trains_path is None:
            self.trains_path = f"{self.logs_dir}/trains.log"
        if self.errors_path is None:
            self.errors_path = f"{self.logs_dir}/errors.log"


@dataclass
class VideoConfig:
    """Base video recording configuration."""

    record_video: bool = False
    video_freq: int = 1000
    video_length: int = 1000
    videos_dir: str = "videos"

    def get_trains_dir(self) -> str:
        return f"{self.videos_dir}/trains"

    def get_evals_dir(self) -> str:
        return f"{self.videos_dir}/evals"

    def get_best_dir(self) -> str:
        return f"{self.videos_dir}/best"


@dataclass
class TrainingVideoConfig(VideoConfig):
    """Training video recording configuration."""

    record_training: bool = False

    @property
    def trains_dir(self) -> str:
        return self.get_trains_dir()

    def __post_init__(self):
        if self.record_training and not self.record_video:
            self.record_video = True


@dataclass
class EvalVideoConfig(VideoConfig):
    """Evaluation video recording configuration."""

    record_eval: bool = False
    record_best: bool = False

    @property
    def evals_dir(self) -> str:
        return self.get_evals_dir()

    @property
    def best_dir(self) -> str:
        return self.get_best_dir()

    def __post_init__(self):
        if self.record_eval and not self.record_video:
            self.record_video = True
        if self.record_best and not self.record_video:
            self.record_video = True


@dataclass
class DeviceConfig:
    """Device and performance configuration."""

    device: str = "auto"  # "auto", "cpu", "cuda"


@dataclass
class FullConfig:
    """Complete configuration combining all sub-configurations."""

    environment: EnvironmentConfig
    feature_extractor: FeatureExtractorConfig
    sac: SACConfig
    network: NetworkConfig
    training: TrainingConfig
    eval: EvalConfig
    checkpoint: CheckPointConfig
    logging: LoggingConfig
    training_video: TrainingVideoConfig
    eval_video: EvalVideoConfig
    device: DeviceConfig
    eval: EvalConfig
    checkpoint: CheckPointConfig
    logging: LoggingConfig
    device: DeviceConfig


# =============================================================================
# Predefined Configuration Presets
# =============================================================================


def get_standard_config() -> FullConfig:
    """
    Standard configuration for normal training.
    Balanced performance and training time.
    """
    return FullConfig(
        environment=EnvironmentConfig(
            max_obstacles=10,
            target_radius=1,
            recognition_radius=5,
            destruction_radius=10,
            n_obstacles=10,
            render_mode="rgb_array",
        ),
        feature_extractor=FeatureExtractorConfig(
            d_model=32,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            include_acceleration=True,
        ),
        sac=SACConfig(
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=128,
            learning_starts=10_000,
        ),
        network=NetworkConfig(net_arch=[32, 16, 16, 8]),
        training=TrainingConfig(
            total_timesteps=1_000_000_000,
            log_interval=100,
            n_envs=8,
        ),
        eval=EvalConfig(
            eval_freq=1_000,
            eval_episodes=4,
            n_eval_envs=4,
            eval_dir="standard/evals",
        ),
        checkpoint=CheckPointConfig(
            save_freq=10_000,
            max_checkpoints=5,
            checkpoint_dir="standard/cps",
        ),
        logging=LoggingConfig(
            logs_dir="standard/logs",
        ),
        training_video=TrainingVideoConfig(
            videos_dir="standard/videos",
            record_training=False,
        ),
        eval_video=EvalVideoConfig(
            videos_dir="standard/videos",
            record_eval=True,
            record_best=True,
            video_freq=1_000,
            video_length=1_000,
        ),
        device=DeviceConfig(device="auto"),
    )


CONFIG_PRESETS = {
    "standard": get_standard_config,
}


def get_config(preset_name: str) -> FullConfig:
    """
    Get a predefined configuration by name.

    Args:
        preset_name: Name of the preset configuration

    Returns:
        FullConfig: Complete configuration object
    """
    if preset_name not in CONFIG_PRESETS:
        available = ", ".join(CONFIG_PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available: {available}"
        )

    return CONFIG_PRESETS[preset_name]()


def list_available_presets() -> List[str]:
    """Get list of available configuration preset names."""
    return list(CONFIG_PRESETS.keys())
