import random
import yaml
import os


class SnakeEngine:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.map_width = self.config["env"]["base_width"]
        self.map_height = self.config["env"]["base_height"]

        self.snake = [(self.map_width // 2, self.map_height // 2)]
        self.direction = (1, 0)
        self.score = 0
        self.step_counter = 0

        self.red_apple = (*self._get_random_empty_pos(), True)
        self.green_apple = (0, 0, False)  # Na start nieaktywne

        return self.get_game_state()

    def _get_random_empty_pos(self):
        while True:
            x = random.randint(0, self.map_width - 1)
            y = random.randint(0, self.map_height - 1)
            if (x, y) not in self.snake:
                return x, y

    def generate_food(self):
        if not self.red_apple[2]:
            self.red_apple = (*self._get_random_empty_pos(), True)

        prob = self.config["env"]["green_apple_prob"]
        if not self.green_apple[2] and random.random() < prob:
            self.green_apple = (*self._get_random_empty_pos(), True)

    def extend_map(self):
        self.map_width += 1
        self.map_height += 1

    def step(self, action):
        self.step_counter += 1

        reward = self.config["rewards"]["life_penalty"]
        game_over = False

        # 0:up, 1:down, 2:left, 3:right
        directions = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        new_dir = directions[action]

        if (new_dir[0] + self.direction[0] == 0) and (
            new_dir[1] + self.direction[1] == 0
        ):
            return self.config["rewards"]["death"], True, self.score

        self.direction = new_dir

        curr_head = self.snake[0]
        new_head = (curr_head[0] + self.direction[0], curr_head[1] + self.direction[1])

        if not (
            0 <= new_head[0] < self.map_width and 0 <= new_head[1] < self.map_height
        ):
            return self.config["rewards"]["death"], True, self.score

        if new_head in self.snake:
            return self.config["rewards"]["death"], True, self.score

        self.snake.insert(0, new_head)

        ate_food = False

        if self.red_apple[2] and new_head == (self.red_apple[0], self.red_apple[1]):
            reward = self.config["rewards"]["red_apple"]
            self.score += 1
            self.red_apple = (0, 0, False)
            ate_food = True

        elif self.green_apple[2] and new_head == (
            self.green_apple[0],
            self.green_apple[1],
        ):
            reward = self.config["rewards"]["green_apple"]
            self.score += 2
            self.green_apple = (0, 0, False)
            ate_food = True

        if ate_food:
            self.generate_food()
            if self.score % self.config["env"]["expand_every"] == 0:
                self.extend_map()
        else:
            self.snake.pop()

        return reward, game_over, self.score

    def get_game_state(self):
        return {
            "snake": self.snake,
            "direction": self.direction,
            "red_apple": self.red_apple,
            "green_apple": self.green_apple,
            "map_width": self.map_width,
            "map_height": self.map_height,
        }


if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "game_config.yaml"
    )

    with open(config_path, "r") as file:
        game_config = yaml.safe_load(file)

    engine = SnakeEngine(game_config)

    print("Initial state:")
    print(engine.get_game_state())

    reward, is_over, score = engine.step(3)

    print(f"\nAfter moving right:")
    print(f"Reward: {reward}, Game Over: {is_over}, Score: {score}")
    print("Snake state:", engine.snake)
