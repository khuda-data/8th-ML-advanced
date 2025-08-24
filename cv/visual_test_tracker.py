import sys
import os
import pygame
import numpy as np
import cv2

# --- [핵심 수정] 프로젝트 루트 디렉토리 경로 설정 변경 ---
# __file__은 현재 파일의 경로(/.../KHUDAFINDER/cv/visual_test_tracker.py)입니다.
# os.path.dirname()을 한 번 쓰면 'cv' 폴더 경로가 나오고,
# 한 번 더 쓰면 우리가 원하는 프로젝트 루트('KHUDAFINDER') 경로가 나옵니다.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 이제 프로젝트 루트가 경로에 포함되었으므로, 다른 모듈을 불러올 수 있습니다.
from env.kf_env import KFEnv
from env.entities import Agent, Entity
from cv import YOLOv8Tracker # 'from cv.tracker import ...' 가 아닌 'from .' 으로 시작하면 안됩니다.
from cv.utils.visualizer import draw_tracked_objects

def run_visual_test():
    """
    CV 모듈의 객체 추적 기능을 시각적으로 테스트하는 메인 함수.
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