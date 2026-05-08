import yaml
import os
import time
from env.env_snake import SnakeEnv

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

    obs, info = env.reset()
    print(f"Observation space shape: {obs.shape} (Expected: (17,))")
    
    if obs.shape != (17,):
        print("ERROR: Wrong observation shape! Check state builder.")
        exit()


    print("\nInitializing test...\n")
    
    episodes = 3 
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        
        print(f"--- Game #{ep + 1} ---")
        
        while not done:
            step_count += 1
            
            action = env.action_space.sample() 
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step: {step_count:02d} | Action: {action} | Reward: {reward:6.2f} | Score: {info['score']}")
            
            done = terminated or truncated
            
            time.sleep(0.05)
            
        print(f">>> END OF GAME! The snake survived {step_count} steps. Final reward (penalty for death): {reward}\n")
        
    print("Test environment completed successfully!")