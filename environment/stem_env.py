import gymnasium as gym
from gymnasium import spaces
import numpy as np

class STEMEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_size = 10
        self.max_steps = 200
        
        # Action space
        self.action_space = spaces.Discrete(8)
        
        # Observation
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0, 5, (self.grid_size, self.grid_size)),
            "stats": spaces.Box(0, 20, (3,))  # students, resources, dropouts
        })
        
        # Rewards
        self.reward_dict = {
            "student": 50,
            "resource": 15,
             "barrier": -5,
            "dropout": -20,
            "step": -0.1,
            "progress": 0.5
        }

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.agent_pos = np.array([self.grid_size//2, self.grid_size//2])
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
        
        # Entity placement
        self._place_entities(2, 3)  # Students
        self._place_entities(3, 2)  # Barriers
        self._place_entities(4, 4)  # Resources
        self._place_entities(5, 2)  # Dropouts
        
        self.steps = 0
        self.students = 0
        self.resources = 0
        self.dropouts = 0
        
        return self._get_obs(), {}

    def step(self, action):
        move = np.array([
            [-1,0], [1,0], [0,-1], [0,1],
            [-1,-1], [-1,1], [1,-1], [1,1]
        ][action])
        
        new_pos = self.agent_pos + move
        reward = self.reward_dict["step"]
        terminated = False
        
        # Progress reward, distance to nearest student
        student_positions = np.argwhere(self.grid == 2)
        if len(student_positions) > 0:
            distances = np.linalg.norm(student_positions - self.agent_pos, axis=1)
            reward += self.reward_dict["progress"] / (1 + np.min(distances))
        
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size):
            
            cell_content = self.grid[new_pos[0], new_pos[1]]
            
            # Update grid
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 0
            self.agent_pos = new_pos
            self.grid[new_pos[0], new_pos[1]] = 1
            
            # Calculate rewards
            if cell_content == 2: 
                reward += self.reward_dict["student"]
                self.students += 1
            elif cell_content == 4: 
                reward += self.reward_dict["resource"]
                self.resources += 1
            elif cell_content == 3: 
                reward += self.reward_dict["barrier"]
            elif cell_content == 5: 
                reward += self.reward_dict["dropout"]
                self.dropouts += 1
        
        self.steps += 1
        terminated = self.steps >= self.max_steps or self.students >= 3
        
        return self._get_obs(), reward, terminated, False, self._get_info()

    def _place_entities(self, entity_type, count):
        for _ in range(count):
            while True:
                x, y = self.np_random.integers(0, self.grid_size, 2)
                if self.grid[x, y] == 0:
                    self.grid[x, y] = entity_type
                    break

    def _get_obs(self):
        return {
            "grid": self.grid.copy(),
            "stats": np.array([self.students, self.resources, self.dropouts])
        }

    def _get_info(self):
        return {
            "students": self.students,
            "resources": self.resources,
            "dropouts": self.dropouts,
            "steps": self.steps
        }
