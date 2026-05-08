import yaml
import os
import time
from env.env_snake import SnakeEnv
from agents.models.dqn_agent import DQNAgent
import numpy as np
import datetime

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'game_config.yaml')
    
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        exit()


    env = SnakeEnv(config)
    agent = DQNAgent(config)

    obs, info = env.reset()
    print(f"Observation space shape: {obs.shape} (Expected: (17,))")
    
    if obs.shape != (17,):
        print("ERROR: Wrong observation shape! Check state builder.")
        exit()


    print("\nInitializing test...\n")
    
    episodes = 500
    best_score = 0
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        
        print(f"--- Game #{ep + 1} ---")
        show_video = (ep == 0 or ep % 10 == 0)
        while not done:
            if show_video:
                env.render()
            step_count += 1
            
            action = agent.get_action(obs)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # print(f"Step: {step_count:02d} | Action: {action} | Reward: {reward:6.2f} | Score: {info['score']}")
            
            done = terminated or truncated
            agent.memory.push(obs, action, reward, next_obs, done)
            agent.learn()
            obs = next_obs
            
            # time.sleep(0.05)
            
        print(f">>> END OF GAME! The snake survived {step_count} steps. Total score: {info['score']}\n")
        if info['score'] > best_score*2:
            best_score = info['score']
            agent.model.save(f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_dqn_snake_model.pth") 
    print("Test environment completed successfully!")

    


