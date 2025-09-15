import numpy 
import gymnasium as gym
from gymnasium import spaces

if not hasattr(numpy, "bool8"):
    numpy.bool8 = numpy.bool_

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, size=5, max_steps=None, render_mode=None):
        super().__init__()
        self.size = int(size)
        self.max_steps = max_steps or (4*self.size)  
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0, high=self.size-1, shape=(2,), dtype=numpy.int32
        )

        self.action_space = spaces.Discrete(4)

        self.start = numpy.array([0, 0], dtype=numpy.int32)
        self.goal  = numpy.array([self.size-1, self.size-1], dtype=numpy.int32)

        self._state = None
        self._steps = 0

    def _get_obs(self):
        return self._state.copy()

    def _get_info(self):
        dist = numpy.abs(self.goal - self._state).sum()
        return {"manhattan_distance": int(dist), "steps": self._steps}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state = self.start.copy()
        self._steps = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        self._steps += 1

        if action == 0:   # up
            self._state[0] = max(0, self._state[0]-1)
        elif action == 1: # right
            self._state[1] = min(self.size-1, self._state[1]+1)
        elif action == 2: # down
            self._state[0] = min(self.size-1, self._state[0]+1)
        elif action == 3: # left
            self._state[1] = max(0, self._state[1]-1)
        else:
            raise ValueError("Invalid action")

        terminated = numpy.array_equal(self._state, self.goal)
        reward = 10.0 if terminated else -1.0

        truncated = self._steps >= self.max_steps
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        grid = [[" . "]*self.size for _ in range(self.size)]
        r, c = self._state
        gr, gc = self.goal
        grid[gr][gc] = " G "
        grid[r][c] = " A "
        lines = ["".join(row) for row in grid]
        frame = "\n".join(lines)
        if self.render_mode == "ansi":
            return frame
        elif self.render_mode == "human":
            print(frame)

    def close(self):
        pass
