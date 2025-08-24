#!/usr/bin/env python3
"""Test script to verify rendering and collision fixes"""

import pygame
import numpy as np
import time
from env.kf_env import KFEnv


def test_rendering_and_collision():
    """Test the rendering and collision fixes"""
    # Create environment with human rendering
    env = KFEnv(
        render_mode="human",
        max_obstacles=5,
        recognition_radius=3.0,
        destruction_radius=10.0,
        target_radius=0.3,
    )

    print("Testing environment rendering and collision fixes...")
    print("- Yellow target (was green)")
    print("- All obstacles visible (not just within recognition radius)")
    print("- Collision prevention (no passthrough)")
    print("- Press ESC to exit, SPACE to reset")

    obs, _ = env.reset()

    # Add some obstacles
    for i in range(3):
        env.add_obstacle(radius=0.8, speed=1.5)

    running = True
    clock = pygame.time.Clock()
    steps = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    obs, _ = env.reset()
                    steps = 0
                    print("Environment reset")

        # Apply random action
        action = np.random.uniform(-1, 1, 2)
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()
        steps += 1

        if terminated or truncated:
            print(f"Episode ended after {steps} steps")
            if env.collision_occurred:
                print("- Collision detected!")
            if env._check_target_reached():
                print("- Target reached!")
            obs, _ = env.reset()
            steps = 0

        clock.tick(60)  # 60 FPS

    env.close()
    print("Test completed!")


if __name__ == "__main__":
    test_rendering_and_collision()
