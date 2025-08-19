#!/usr/bin/env python3
"""
Test script for the newly refactored KFEnv with Dict observation space
"""

import numpy as np
from env import KFEnv
from env.entities import Agent, Entity
from env.types import Vector2D


def test_new_env():
    """Test the new KFEnv with Dict observation space and 7D entity representation"""

    # Create environment with max_obstacles
    env = KFEnv(max_obstacles=5)

    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Create and add agent (no position needed - will be randomly set in reset)
    agent = Agent(radius=0.5)
    env.add_agent(agent)

    # Create and add some obstacles (no position needed - will be randomly set in reset)
    for i in range(3):
        obstacle = Entity(radius=0.3, mass=1.0)
        env.add_obstacle(obstacle)

    print(f"Added agent and {len(env.obstacles)} obstacles")

    # Reset environment
    obs, info = env.reset()

    print("\n=== Initial Observation ===")
    print(f"Agent shape: {obs['agent'].shape}, values: {obs['agent']}")
    print(f"Obstacles shape: {obs['obstacles'].shape}")
    print(f"Target shape: {obs['target'].shape}, values: {obs['target']}")
    print(f"Mask shape: {obs['mask'].shape}, values: {obs['mask']}")

    # Test a few steps
    print("\n=== Testing Steps ===")
    for step in range(3):
        action = np.random.uniform(-1, 1, 2)  # Random action
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step + 1}:")
        print(f"  Action: {action}")
        print(
            f"  Agent position: [{obs['agent'][1]:.3f}, {obs['agent'][2]:.3f}]"
        )
        print(
            f"  Agent velocity: [{obs['agent'][3]:.3f}, {obs['agent'][4]:.3f}]"
        )
        print(
            f"  Agent acceleration: [{obs['agent'][5]:.3f}, {obs['agent'][6]:.3f}]"
        )
        print(f"  Reward: {reward:.3f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"  Collision flag: {env.collision_occurred}")

        if terminated or truncated:
            break

    # Test observation space consistency
    print("\n=== Testing Observation Space Consistency ===")
    for key, space in env.observation_space.spaces.items():
        obs_value = obs[key]
        if space.contains(obs_value):
            print(f"✓ {key}: observation fits in space")
        else:
            print(f"✗ {key}: observation does NOT fit in space")
            print(f"  Space: {space}")
            print(f"  Observation shape: {obs_value.shape}")

    # Test collision detection with pymunk handlers
    print("\n=== Testing Collision Detection ===")
    env.reset()
    # Try to cause a collision by moving towards obstacles
    collision_detected = False
    for step in range(100):
        # Move towards first obstacle if it exists
        if len(env.obstacles) > 0:
            agent_pos = env.agent.get_position()
            obstacle_pos = env.obstacles[0].get_position()

            # Calculate direction towards obstacle
            dx = obstacle_pos.x - agent_pos.x
            dy = obstacle_pos.y - agent_pos.y
            length = np.sqrt(dx * dx + dy * dy)

            if length > 0:
                action = np.array(
                    [dx / length, dy / length]
                )  # Normalized direction
            else:
                action = np.array([1.0, 0.0])
        else:
            action = np.array([1.0, 0.0])

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print(f"✓ Collision detected at step {step + 1}")
            print(f"  Collision flag: {env.collision_occurred}")
            collision_detected = True
            break

    if not collision_detected:
        print("? No collision detected in 100 steps")

    # Test maximum obstacles
    print("\n=== Testing Maximum Obstacles ===")
    try:
        for i in range(10):  # Try to add more than max_obstacles
            obstacle = Entity(radius=0.2)
            env.add_obstacle(obstacle)
    except ValueError as e:
        print(f"✓ Correctly caught max obstacles limit: {e}")

    # Test out of bounds (truncated)
    print("\n=== Testing Out of Bounds ===")
    env.reset()
    # Try to move agent far out of bounds
    for _ in range(50):
        action = np.array([1.0, 1.0])  # Max action towards corner
        obs, reward, terminated, truncated, info = env.step(action)
        if truncated:
            print("✓ Agent went out of bounds and episode was truncated")
            break
    else:
        print("? Agent did not go out of bounds in 50 steps")

    env.close()
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_new_env()
