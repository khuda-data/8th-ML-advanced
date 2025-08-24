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

    max_obstacles: int = 10
    target_radius: float = 10.0
    recognition_radius: float = 100.0
    destruction_radius: float = 200.0
    n_obstacles: int = 5
    render_mode: Optional[str] = None  # None, "human", "rgb_array"


@dataclass
class FeatureExtractorConfig:
    """Feature extractor (attention mechanism) configuration."""

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    include_acceleration: bool = False
    features_dim: int = 128


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
    n_envs: int = 4  # Number of parallel training environments
    n_eval_envs: int = 1  # Number of parallel evaluation environments
    seed: int = 42


@dataclass
class LoggingConfig:
    """Logging and saving configuration."""

    save_path: str = "sac_attention"
    tensorboard_log: str = None  # Will be auto-generated from save_path
    save_freq: int = 50_000
    verbose: int = 1

    def __post_init__(self):
        # Auto-generate tensorboard_log path if not provided
        if self.tensorboard_log is None:
            self.tensorboard_log = f"{self.save_path}/logs/"


@dataclass
class VideoConfig:
    """Video recording configuration."""

    record_video: bool = True
    video_freq: int = 50_000  # Video recording frequency
    video_length: int = 1000  # Length of recorded videos


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


def get_quick_test_config() -> FullConfig:
    """
    Quick test configuration for development and debugging.
    Fast training with minimal resources.
    """
    return FullConfig(
        environment=EnvironmentConfig(
            max_obstacles=5,
            target_radius=10.0,
            recognition_radius=50.0,
            destruction_radius=100.0,
            n_obstacles=3,
            render_mode=None,
        ),
        feature_extractor=FeatureExtractorConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            dropout=0.1,
            include_acceleration=False,
            features_dim=64,
        ),
        sac=SACConfig(
            learning_rate=1e-3,
            buffer_size=50_000,
            batch_size=128,
            learning_starts=1_000,
        ),
        network=NetworkConfig(net_arch=[128, 128]),
        training=TrainingConfig(
            total_timesteps=10_000,
            eval_freq=2_000,
            eval_episodes=3,
            log_interval=500,
            n_envs=2,
            n_eval_envs=1,
        ),
        logging=LoggingConfig(save_freq=5_000, save_path="quick_test"),
        video=VideoConfig(
            record_video=True, video_freq=5_000, video_length=500
        ),
        device=DeviceConfig(device="auto"),
    )


def get_standard_config() -> FullConfig:
    """
    Standard configuration for normal training.
    Balanced performance and training time.
    """
    return FullConfig(
        environment=EnvironmentConfig(
            max_obstacles=10,
            target_radius=10.0,
            recognition_radius=100.0,
            destruction_radius=200.0,
            n_obstacles=5,
            render_mode=None,
        ),
        feature_extractor=FeatureExtractorConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            include_acceleration=False,
            features_dim=128,
        ),
        sac=SACConfig(
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            learning_starts=10_000,
        ),
        network=NetworkConfig(net_arch=[256, 256]),
        training=TrainingConfig(
            total_timesteps=1_000_000,
            eval_freq=20_000,
            eval_episodes=5,
            log_interval=1000,
            n_envs=4,
            n_eval_envs=1,
        ),
        logging=LoggingConfig(save_freq=50_000, save_path="standard"),
        video=VideoConfig(
            record_video=True, video_freq=50_000, video_length=1000
        ),
        device=DeviceConfig(device="auto"),
    )


def get_high_performance_config() -> FullConfig:
    """
    High-performance configuration for serious training.
    Larger networks, more environments, longer training.
    """
    return FullConfig(
        environment=EnvironmentConfig(
            max_obstacles=15,
            target_radius=8.0,
            recognition_radius=120.0,
            destruction_radius=250.0,
            n_obstacles=8,
            render_mode=None,
        ),
        feature_extractor=FeatureExtractorConfig(
            d_model=128,
            n_heads=8,
            n_layers=3,
            dropout=0.1,
            include_acceleration=True,
            features_dim=256,
        ),
        sac=SACConfig(
            learning_rate=1e-4,
            buffer_size=2_000_000,
            batch_size=512,
            learning_starts=25_000,
        ),
        network=NetworkConfig(net_arch=[512, 512, 256]),
        training=TrainingConfig(
            total_timesteps=5_000_000,
            eval_freq=50_000,
            eval_episodes=10,
            log_interval=2000,
            n_envs=8,
            n_eval_envs=2,
        ),
        logging=LoggingConfig(save_freq=100_000, save_path="high_performance"),
        video=VideoConfig(
            record_video=True, video_freq=100_000, video_length=1500
        ),
        device=DeviceConfig(device="cuda"),
    )


def get_minimal_config() -> FullConfig:
    """
    Minimal configuration for resource-constrained environments.
    Smallest possible setup while maintaining functionality.
    """
    return FullConfig(
        environment=EnvironmentConfig(
            max_obstacles=5,
            target_radius=15.0,
            recognition_radius=80.0,
            destruction_radius=150.0,
            n_obstacles=3,
            render_mode=None,
        ),
        feature_extractor=FeatureExtractorConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
            include_acceleration=False,
            features_dim=64,
        ),
        sac=SACConfig(
            learning_rate=5e-4,
            buffer_size=100_000,
            batch_size=64,
            learning_starts=5_000,
        ),
        network=NetworkConfig(net_arch=[128, 64]),
        training=TrainingConfig(
            total_timesteps=100_000,
            eval_freq=10_000,
            eval_episodes=3,
            log_interval=1000,
            n_envs=2,
            n_eval_envs=1,
        ),
        logging=LoggingConfig(save_freq=25_000, save_path="minimal"),
        video=VideoConfig(
            record_video=False, video_freq=25_000, video_length=300
        ),
        device=DeviceConfig(device="cpu"),
    )


# =============================================================================
# Configuration Registry
# =============================================================================

CONFIG_PRESETS = {
    "quick_test": get_quick_test_config,
    "standard": get_standard_config,
    "high_performance": get_high_performance_config,
    "minimal": get_minimal_config,
}


def get_config(preset_name: str) -> FullConfig:
    """
    Get a predefined configuration by name.

    Args:
        preset_name: Name of the preset configuration

    Returns:
        FullConfig: Complete configuration object

    Available presets:
        - "quick_test": Fast testing configuration
        - "standard": Balanced training configuration
        - "high_performance": Resource-intensive optimal training
        - "attention_research": Focus on attention mechanism research
        - "minimal": Minimal resource configuration
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


def print_config_summary(config: FullConfig) -> None:
    """Print a summary of the configuration."""
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)

    print(
        f"Environment: {config.environment.n_obstacles}/{config.environment.max_obstacles} obstacles"
    )
    print(
        f"  Radii: target={config.environment.target_radius}, recognition={config.environment.recognition_radius}"
    )

    print(
        f"Feature Extractor: d_model={config.feature_extractor.d_model}, heads={config.feature_extractor.n_heads}"
    )
    print(
        f"  Layers={config.feature_extractor.n_layers}, features_dim={config.feature_extractor.features_dim}"
    )

    print(
        f"SAC: lr={config.sac.learning_rate}, batch_size={config.sac.batch_size}"
    )
    print(
        f"  Buffer={config.sac.buffer_size:,}, learning_starts={config.sac.learning_starts:,}"
    )

    print(f"Network: {config.network.net_arch}")

    print(f"Training: {config.training.total_timesteps:,} timesteps")
    print(
        f"  Parallel envs: {config.training.n_envs} training, {config.training.n_eval_envs} eval"
    )

    print(f"Video: {'enabled' if config.video.record_video else 'disabled'}")
    if config.video.record_video:
        print(
            f"  Frequency: every {config.video.video_freq:,} steps, length={config.video.video_length}"
        )

    print(f"Device: {config.device.device}")
    print("=" * 60)
