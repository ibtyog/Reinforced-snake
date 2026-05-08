import random
import yaml
import os
import pygame


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
            if len(self.snake) >= self.map_width * self.map_height // 2:
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
    
    def render(self):
        MAX_WINDOW_SIZE = 800 

        if not hasattr(self, 'window') or self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((MAX_WINDOW_SIZE, MAX_WINDOW_SIZE))
            pygame.display.set_caption("Reinforced Snake")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.window.fill((20, 20, 20))

        max_map_dimension = max(self.map_width, self.map_height)
        
        cell_size = int(MAX_WINDOW_SIZE / max_map_dimension)

        if cell_size < 1:
            cell_size = 1

        pixel_map_width = self.map_width * cell_size
        pixel_map_height = self.map_height * cell_size
        
        offset_x = (MAX_WINDOW_SIZE - pixel_map_width) // 2
        offset_y = (MAX_WINDOW_SIZE - pixel_map_height) // 2

        pygame.draw.rect(self.window, (30, 30, 30), 
                         (offset_x, offset_y, pixel_map_width, pixel_map_height))

        rx, ry, r_active = self.red_apple
        if r_active:
            pygame.draw.rect(self.window, (255, 50, 50), 
                             (offset_x + rx * cell_size, offset_y + ry * cell_size, cell_size, cell_size))

        gx, gy, g_active = self.green_apple
        if g_active:
            pygame.draw.rect(self.window, (50, 255, 50), 
                             (offset_x + gx * cell_size, offset_y + gy * cell_size, cell_size, cell_size))

        for i, (x, y) in enumerate(self.snake):
            color = (0, 200, 255) if i == 0 else (0, 100, 200)
            rect = (offset_x + x * cell_size, offset_y + y * cell_size, cell_size, cell_size)
            pygame.draw.rect(self.window, color, rect)
            
            if cell_size > 4:
                pygame.draw.rect(self.window, (0, 0, 0), rect, 1)

        pygame.display.flip()
        
        self.clock.tick(15)



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
