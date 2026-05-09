import wandb


class WandbLogger:
    """
    MLOps integration module for Weights & Biases (W&B).

    Asynchronously logs training metrics (scores, total rewards, epsilon decay)
    to the cloud. Allows for real-time monitoring, visualization, and
    comparison of different experiments via a web dashboard.
    """

    def __init__(self, config):
        self.use_wandb = config["wandb"].get("use_wandb", False)

        if self.use_wandb:
            # Inicjalizacja projektu w W&B
            wandb.init(
                project=config["wandb"].get("project_name", "Snake-RL"),
                name=config["wandb"].get("run_name", "DQN_Run"),
                config=config,
            )
            print("Weights & Biases initialized.")
        else:
            print(
                "W&B is disabled. To enable, set 'use_wandb' to True in the config and provide 'project_name' and 'run_name'."
            )

    def log_episode(self, episode, score, total_reward, epsilon, steps):
        if self.use_wandb:
            wandb.log(
                {
                    "Episode": episode,
                    "Score": score,
                    "Total Reward": total_reward,
                    "Epsilon": epsilon,
                    "Steps survived": steps,
                }
            )

    def finish(self):
        if self.use_wandb:
            wandb.finish()
