import os
import yaml
from env.snake_env import SnakeEnv
from agents.dqn_agent import DQNAgent
from visualisation.renderer import PygameRenderer
from training.logger import WandbLogger as Logger
import re


class Trainer:
    """
    Main orchestrator of the Machine Learning training pipeline.

    Initializes the environment, agent, loggers, and renderer based on
    YAML configurations. Manages the main training loop (episodes and steps),
    handles data flow between the environment and the agent, and controls
    the checkpointing (model save/resume) system.
    """

    def __init__(self):
        self.config = self._load_all_configs()
        self.renderer = PygameRenderer()
        self.logger = Logger(self.config)
        self.env = SnakeEnv(self.config)
        self.agent = DQNAgent(self.config)

        self.episodes = self.config["training"]["episodes"]
        self.render_every = self.config["training"]["render_every"]
        self.best_score = 0

        resume = self.config["training"].get("resume_training", False)
        if resume:
            model_name = self.config["training"].get("resume_model_path", "")
            model_path = os.path.join("saved_models", model_name)
            if os.path.exists(model_path):
                print(f"Resuming training from model: {model_name}")
                self.agent.load_model(model_path, is_training=True)
                match = re.search(r"best_model_score_(\d+).pth", model_name)
                if match:
                    self.best_score = int(match.group(1))
            else:
                print(
                    f"Model file {model_name} not found. Starting training from scratch."
                )

    def _load_all_configs(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        configs_dir = os.path.join(base_dir, "configs")

        full_config = {}
        files = ["env_config.yaml", "agent_config.yaml", "train_config.yaml"]

        for file_name in files:
            path = os.path.join(configs_dir, file_name)
            with open(path, "r", encoding="utf-8") as f:
                config_part = yaml.safe_load(f)
                full_config.update(config_part)

        return full_config

    def run(self):

        for ep in range(self.episodes):
            obs, info = self.env.reset()
            done = False
            step_count = 0
            total_reward = 0.0

            show_video = (ep == 0 or (ep + 1) % self.render_every == 0) and self.config[
                "training"
            ]["render_video"]

            while not done:
                if show_video:
                    stats = {
                        "episode": ep + 1,
                        "score": info.get("score", 0),
                        "best_score": self.best_score,
                        "epsilon": self.agent.epsilon,
                        "total_reward": total_reward,
                    }
                    self.renderer.render(self.env.snake_engine, stats)

                step_count += 1

                action = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                total_reward += reward

                self.agent.memory.push(obs, action, reward, next_obs, done)
                self.agent.learn()

                obs = next_obs
            if ep % 100 == 0 or ep == self.episodes - 1:
                print(
                    f"Game: {ep + 1:04d}/{self.episodes} | Score: {info.get('score', 0):02d} | Steps: {step_count:03d} | Epsilon: {self.agent.epsilon:.3f}"
                )
            self.logger.log_episode(
                episode=ep + 1,
                score=info.get("score", 0),
                total_reward=total_reward,
                epsilon=self.agent.epsilon,
                steps=step_count,
            )

            if info.get("score", 0) > self.best_score:
                self.best_score = info.get("score", 0)
                print(f"New Best Score: {self.best_score}.")
                self.agent.model.save(f"best_model_score_{self.best_score}.pth")
        self.logger.finish()
