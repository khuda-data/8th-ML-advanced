from enum import IntEnum, Enum


class CollisionType(IntEnum):
    ENTITY = 1
    AGENT = 2
    OBSTACLE = 3
    WALL = 4


class RewardType(float, Enum):
    # Distance-based reward parameters
    TARGET_REWARD_SCALE = 30.0  # Scale for target proximity reward
    OBSTACLE_PENALTY_SCALE = 50.0  # Scale for obstacle avoidance penalty
    SAFETY_RADIUS = 3.5  # Minimum safe distance from obstacles
    TARGET_BONUS = 100.0  # Bonus for reaching target
    COLLISION_PENALTY = 100.0  # Penalty for collision
    BOUNDARY_PENALTY = 100.0  # Penalty for going out of bounds
    TIME_PENALTY = 0.1
    STOP_PENALTY = 0.3