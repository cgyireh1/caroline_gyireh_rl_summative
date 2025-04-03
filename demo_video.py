import pygame
import cv2
import numpy as np
import torch
from environment.stem_env import STEMEnv
from training.pg_training import PolicyNetwork
from stable_baselines3 import DQN
import os

def record_demo(model_path, output_path="demo.mp4", model_type='pg', fps=10, episodes=3):
    env = STEMEnv()
    pygame.init()
    font = pygame.font.SysFont('Arial', 24)
    
    # Load model
    if model_type == 'pg':
        model = PolicyNetwork(env.observation_space, env.action_space)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        model = DQN.load(model_path)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (800, 800))
    
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            screen = pygame.Surface((800, 800))
            
            # Get action
            if model_type == 'pg':
                with torch.no_grad():
                    action_probs = model(obs)
                    action = torch.argmax(action_probs).item()
                    confidence = torch.max(action_probs).item()
            else:
                action, _ = model.predict(obs, deterministic=True)
                confidence = 1.0
            
            # Step environment
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            
            # Render grid
            cell_size = 80
            for x in range(env.grid_size):
                for y in range(env.grid_size):
                    color = [
                        (240, 240, 240),  # Empty
                        (50, 130, 200),   # Agent
                        (230, 150, 230),  # Student
                        (200, 50, 50),    # Barrier
                        (100, 200, 100),  # Resource
                        (100, 100, 100)   # Dropout
                    ][env.grid[x, y]]
                    pygame.draw.rect(
                        screen, color,
                        (x*cell_size, y*cell_size, cell_size, cell_size)
                    )
            
            # Display stats
            stats = [
                f"Episode: {episode+1}/{episodes}",
                f"Step: {env.steps}",
                f"Total Reward: {total_reward:.1f}",
                f"Students: {info['students']}",
                f"Resources: {info['resources']}",
                f"Dropouts: {info['dropouts']}",
                f"Confidence: {confidence:.2f}"
            ]
            
            for i, stat in enumerate(stats):
                text = font.render(stat, True, (0, 0, 0))
                screen.blit(text, (10, 10 + i*25))
            
            # Save frame
            frame = np.transpose(
                pygame.surfarray.array3d(screen),
                axes=(1, 0, 2))
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            pygame.time.delay(int(3000/fps))
            
            if env.steps >= 200:
                break
        
        print(f"Episode {episode+1} completed with reward: {total_reward:.1f}")
    
    video_writer.release()
    pygame.quit()
    print(f"Video saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output', default="demo.mp4")
    parser.add_argument('--model_type', choices=['pg', 'dqn'], default='pg')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--episodes', type=int, default=3)
    args = parser.parse_args()
    
    record_demo(
        model_path=args.model_path,
        output_path=args.output,
        model_type=args.model_type,
        fps=args.fps,
        episodes=args.episodes
    )
