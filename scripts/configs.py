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
    eval_freq: int = 10_000
    eval_episodes: int = 5
    log_interval: int = 1000
    n_envs: int = 8  # Number of parallel training environments
    n_eval_envs: int = 2  # Number of parallel evaluation environments
    seed: int = 42


@dataclass
class LoggingConfig:
    """Logging and saving configuration."""

    save_path: str = "sac_attention"
    tensorboard_log: str = None  # Will be auto-generated from save_path
    save_freq: int = 10000
    verbose: int = 1

    def __post_init__(self):
        # Auto-generate tensorboard_log path if not provided
        if self.tensorboard_log is None:
            self.tensorboard_log = f"{self.save_path}/logs/"


@dataclass
class VideoConfig:
    """Video recording configuration."""

    record_video: bool = True
    video_freq: int = 5000  # Video recording frequency
    video_length: int = 10000  # Length of recorded videos


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
    logging: LoggingConfig
    video: VideoConfig
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
            max_obstacles=50,
            target_radius=1,
            recognition_radius=10,
            destruction_radius=25,
            n_obstacles=50,
            render_mode="rgb_array",
        ),
        feature_extractor=FeatureExtractorConfig(
            d_model=64,
            n_heads=4,
            n_layers=1,
            dropout=0.1,
            include_acceleration=True,
        ),
        sac=SACConfig(
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=128,
            learning_starts=1000,
        ),
        network=NetworkConfig(net_arch=[256, 256]),
        training=TrainingConfig(
            total_timesteps=1_000_000_000,
            eval_freq=100,
            eval_episodes=4,
            log_interval=10,
            n_envs=8,
            n_eval_envs=4,
        ),
        logging=LoggingConfig(save_freq=1_000, save_path="standard"),
        video=VideoConfig(
            record_video=True, video_freq=1_000, video_length=1_000
        ),
        device=DeviceConfig(device="auto"),
    )


# =============================================================================
# Configuration Registry
# =============================================================================

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
