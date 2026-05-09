import pygame


class PygameRenderer:
    """
    Pygame-based visualization module (View in MVC architecture).

    Receives raw physical state data and renders it into a graphical window.
    Handles dynamic camera scaling, object drawing, and generates a
    Head-Up Display (HUD) overlay to show real-time training statistics.
    """

    def __init__(self, max_window_size=800):
        self.max_window_size = max_window_size
        self.window = None
        self.clock = None
        self.font = None

    def _init_pygame(self):
        pygame.init()
        pygame.font.init()
        self.window = pygame.display.set_mode(
            (self.max_window_size, self.max_window_size)
        )
        pygame.display.set_caption("Snake AI - Dashboard")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 20, bold=True)

    def render(self, engine, hud_stats):
        if self.window is None:
            self._init_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.window.fill((20, 20, 20))

        max_map_dimension = max(engine.map_width, engine.map_height)
        cell_size = max(1, int(self.max_window_size / max_map_dimension))

        pixel_map_width = engine.map_width * cell_size
        pixel_map_height = engine.map_height * cell_size
        offset_x = (self.max_window_size - pixel_map_width) // 2
        offset_y = (self.max_window_size - pixel_map_height) // 2

        pygame.draw.rect(
            self.window,
            (30, 30, 30),
            (offset_x, offset_y, pixel_map_width, pixel_map_height),
        )

        rx, ry, r_active = engine.red_apple
        if r_active:
            pygame.draw.rect(
                self.window,
                (255, 50, 50),
                (
                    offset_x + rx * cell_size,
                    offset_y + ry * cell_size,
                    cell_size,
                    cell_size,
                ),
            )

        gx, gy, g_active = engine.green_apple
        if g_active:
            pygame.draw.rect(
                self.window,
                (50, 255, 50),
                (
                    offset_x + gx * cell_size,
                    offset_y + gy * cell_size,
                    cell_size,
                    cell_size,
                ),
            )

        for i, (x, y) in enumerate(engine.snake):
            color = (0, 200, 255) if i == 0 else (0, 100, 200)
            rect = (
                offset_x + x * cell_size,
                offset_y + y * cell_size,
                cell_size,
                cell_size,
            )
            pygame.draw.rect(self.window, color, rect)
            if cell_size > 4:
                pygame.draw.rect(self.window, (0, 0, 0), rect, 1)

        hud_text = f"Episode: {hud_stats['episode']} | Score: {hud_stats['score']} | Best Score: {hud_stats['best_score']} | Epsilon: {hud_stats['epsilon']:.2f} | Total Reward: {hud_stats.get('total_reward', 0):.2f}"

        text_surface = self.font.render(hud_text, True, (255, 255, 255))

        bg_surface = pygame.Surface(
            (text_surface.get_width() + 10, text_surface.get_height() + 10)
        )
        bg_surface.set_alpha(150)
        bg_surface.fill((0, 0, 0))

        self.window.blit(bg_surface, (5, 5))
        self.window.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(15)
