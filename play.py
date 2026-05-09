import os
import yaml
import time
from env.snake_env import SnakeEnv
from agents.dqn_agent import DQNAgent
from visualisation.renderer import PygameRenderer


def load_all_configs():
    base_dir = os.path.dirname(__file__)
    configs_dir = os.path.join(base_dir, "configs")

    full_config = {}
    files = ["env_config.yaml", "agent_config.yaml", "train_config.yaml"]

    for file_name in files:
        path = os.path.join(configs_dir, file_name)
        with open(path, "r", encoding="utf-8") as f:
            full_config.update(yaml.safe_load(f))

    return full_config


if __name__ == "__main__":
    print("Sandbox game, not training.")
    config = load_all_configs()

    env = SnakeEnv(config)
    agent = DQNAgent(config)
    renderer = PygameRenderer()

    MODEL_PATH = os.path.join("saved_models", "best_model_score_67.pth")

    try:
        agent.load_model(MODEL_PATH)
    except FileNotFoundError:
        print(f"ERROR: File not found {MODEL_PATH}!")

    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        total_reward += reward

        stats = {
            "episode": "TEST",
            "score": info.get("score", 0),
            "best_score": "-",
            "epsilon": agent.epsilon,
            "total_reward": total_reward,
        }

        renderer.render(env.snake_engine, stats)

        time.sleep(0.1)

        obs = next_obs

    print(f"\nScore: {info.get('score', 0)} points.")
