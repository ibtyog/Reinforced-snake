import numpy as np


class StateBuilder:
    def __init__(self):
        # TODO: state vector config load
        pass

    def build_state(self, game_state):
        snake = game_state["snake"]
        head_x, head_y = snake[0]
        dir_x, dir_y = game_state["direction"]
        map_w = game_state["map_width"]
        map_h = game_state["map_height"]

        state_array = np.zeros(17, dtype=float)

        # direct danger
        state_array[0] = self._check_collision(
            head_x + dir_x, head_y + dir_y, game_state
        )
        state_array[1] = self._check_collision(
            head_x - dir_y, head_y + dir_x, game_state
        )
        state_array[2] = self._check_collision(
            head_x + dir_y, head_y - dir_x, game_state
        )

        # direction

        state_array[3] = 1 if (dir_x, dir_y) == (0, -1) else 0
        state_array[4] = 1 if (dir_x, dir_y) == (0, 1) else 0
        state_array[5] = 1 if (dir_x, dir_y) == (-1, 0) else 0
        state_array[6] = 1 if (dir_x, dir_y) == (1, 0) else 0

        # red_apple direction + normalized manhattan distance
        self._process_apple(
            game_state["red_apple"],
            head_x,
            head_y,
            map_w,
            map_h,
            state_array,
            base_idx=7,
            dist_idx=15,
        )

        # green_apple direction + normalized manhattan distance
        self._process_apple(
            game_state["green_apple"],
            head_x,
            head_y,
            map_w,
            map_h,
            state_array,
            base_idx=11,
            dist_idx=16,
        )

        return state_array

    def _check_collision(self, x, y, game_state):
        if (
            x < 0
            or x >= game_state["map_width"]
            or y < 0
            or y >= game_state["map_height"]
        ):
            return 1
        if (x, y) in game_state["snake"]:
            return 1
        return 0

    def _process_apple(
        self, apple_data, head_x, head_y, map_w, map_h, state_array, base_idx, dist_idx
    ):
        ax, ay, is_active = apple_data

        if is_active:
            if ax < head_x:
                state_array[base_idx] = 1
            elif ax > head_x:
                state_array[base_idx + 1] = 1

            if ay < head_y:
                state_array[base_idx + 2] = 1
            elif ay > head_y:
                state_array[base_idx + 3] = 1

            dist = abs(ax - head_x) + abs(ay - head_y)
            state_array[dist_idx] = dist / (map_w + map_h)
        else:
            state_array[dist_idx] = 1.0
