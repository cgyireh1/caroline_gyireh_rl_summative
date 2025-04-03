from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from environment.stem_env import STEMEnv
import numpy as np
import os

class MetricsCallback(EvalCallback):
    """EvalCallback to capture metrics"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.episode_rewards.append(self.last_mean_reward)
        return True

def train_dqn():
    # Setup environment and logging
    env = Monitor(STEMEnv())
    os.makedirs("./logs/dqn", exist_ok=True)
    os.makedirs("./models/dqn", exist_ok=True)
    
    # Initialize model
    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=0.0001,
        buffer_size=100000,
        batch_size=128,
        gamma=0.99,
        exploration_fraction=0.5,
        exploration_final_eps=0.01,
        target_update_interval=500,
        train_freq=16,
        gradient_steps=8,
        policy_kwargs=dict(net_arch=[256, 256])
    )
    
    
    eval_callback = MetricsCallback(
        env,
        best_model_save_path="./models/dqn/best",
        log_path="./logs/dqn",
        eval_freq=2000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # Train model
    model.learn(
        total_timesteps=50000,
        callback=eval_callback,
        log_interval=10,
        progress_bar=True
    )
    
    # Save results
    model.save("./models/dqn/final_model")
    np.savez("./logs/dqn/metrics.npz",
             rewards=eval_callback.episode_rewards)
    
    return model
