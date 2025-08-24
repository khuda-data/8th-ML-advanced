import sys
import os
import pygame
import numpy as np
import cv2

# __file__ 현재 파일의 경로(/.../KHUDAFINDER/cv/visual_test_tracker.py)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from env.kf_env import KFEnv
from env.entities import Agent, Entity
from cv import YOLOv8Tracker 
from cv.utils.visualizer import draw_tracked_objects

def run_visual_test():
    """
    CV 모듈의 객체 추적 기능을 시각적으로 테스트하는 메인 함수
    """
    print("CV 모듈 시각적 테스트를 시작합니다...")
    
    env = KFEnv(render_mode='rgb_array', max_obstacles=5)
    env.reset()

    agent = Agent(radius=0.5) 
    env.add_agent(agent)
    for _ in range(5):
        obstacle = Entity(radius=0.7) 
        env.add_obstacle(obstacle)
    env._reset_entities_safely()

    tracker = YOLOv8Tracker('yolov8n.pt') 
    print("YOLOv8Tracker 로딩 완료.")

    running = True
    while running:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        frame_pygame = env.screen
        frame_np = pygame.surfarray.array3d(frame_pygame)
        frame_np = np.transpose(frame_np, (1, 0, 2))
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        tracked_objects = tracker.track(frame_bgr.copy())
        
        vis_frame = draw_tracked_objects(frame_bgr, tracked_objects)

        cv2.imshow("CV Tracker Visual Test (Press 'q' to quit)", vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
        
        if terminated or truncated:
            env.reset()
            agent = Agent(radius=0.5)
            env.add_agent(agent)
            for _ in range(5):
                obstacle = Entity(radius=0.7)
                env.add_obstacle(obstacle)
            env._reset_entities_safely()

    env.close()
    cv2.destroyAllWindows()
    print("테스트를 종료합니다.")

if __name__ == '__main__':
    run_visual_test()