import os
import csv
import torch
import numpy as np
from collections import deque
import torch.optim as optim
from torch.distributions import Categorical
from environment.stem_env import STEMEnv


class PolicyNetwork(torch.nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        grid_size = obs_space['grid'].shape[0] * obs_space['grid'].shape[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(grid_size + 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_space.n)
        )
    
    def forward(self, obs):
        grid = torch.FloatTensor(obs['grid']).flatten()
        stats = torch.FloatTensor(obs['stats'])
        return self.net(torch.cat([grid, stats]))

def train_pg():
    env = STEMEnv()
    model = PolicyNetwork(env.observation_space, env.action_space)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    gamma = 0.99
    entropy_coef = 0.2
    
    # Setup logging
    os.makedirs("./logs/pg", exist_ok=True)
    os.makedirs("./models/pg", exist_ok=True)
    
    with open("./logs/pg/training.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'reward', 'entropy', 'students', 'steps'])
        
        reward_history = deque(maxlen=100)
        best_reward = -np.inf
        
        for episode in range(1000):
            obs, _ = env.reset()
            saved_log_probs = []
            saved_entropies = []
            rewards = []
            done = False
            
            # Run episode
            while not done:
                action_probs = torch.nn.functional.softmax(model(obs), dim=-1)
                dist = Categorical(action_probs)
                action = dist.sample()
                
                obs, reward, done, _, info = env.step(action.item())
                
                saved_log_probs.append(dist.log_prob(action))
                saved_entropies.append(dist.entropy())  # Entropy calculation
                rewards.append(reward)
            
            # Calculate returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-6) * 0.5 # Normalize returns
            
            # Calculate losses
            policy_loss = []
            entropy_loss = []
            for log_prob, R, entropy in zip(saved_log_probs, returns, saved_entropies):
                policy_loss.append(-log_prob * R)
                entropy_loss.append(-entropy)
            
            policy_loss = torch.stack(policy_loss).sum()
            entropy_loss = torch.stack(entropy_loss).sum()
            loss = policy_loss + entropy_coef * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            # Log metrics
            episode_reward = sum(rewards)
            episode_entropy = torch.stack(saved_entropies).mean().item()
            reward_history.append(episode_reward)
            
            writer.writerow([
                episode,
                episode_reward,
                episode_entropy,
                info['students'],
                info['steps']
            ])
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(model.state_dict(), "./models/pg/best_model.pt")
            
            if episode % 10 == 0:
                avg_reward = np.mean(reward_history)
                print(f"Episode {episode}, Avg Reward: {avg_reward:.1f}, Entropy: {episode_entropy:.2f}")
    
    torch.save(model.state_dict(), "./models/pg/final_model.pt")
