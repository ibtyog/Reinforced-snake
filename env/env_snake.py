import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np

from env.engine import SnakeEngine
from env.state_builder import StateBuilder

class SnakeEnv(Env):
    def __init__(self, config):
        super().__init__()

        self.snake_engine = SnakeEngine(config)  
        self.state_builder = StateBuilder()

        self.action_space = spaces.Discrete(4)  
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(17,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        raw_state = self.snake_engine.reset()
        
        observation = self.state_builder.build_state(raw_state)
        
        info = {}
        return np.array(observation, dtype=np.float32), info

    def step(self, action):
        reward, terminated, score = self.snake_engine.step(action)
        
        raw_state = self.snake_engine.get_game_state()
        observation = self.state_builder.build_state(raw_state)
        truncated = False 
        info = {"score": score}
        
        return np.array(observation, dtype=np.float32), reward, terminated, truncated, info
    
    def render(self):
        self.snake_engine.render()