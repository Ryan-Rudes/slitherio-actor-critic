from collections import deque
from gym import Wrapper
import numpy as np
import mahotas

class FrameStack(Wrapper):
    def __init__(self, env):
        super(FrameStack, self).__init__(env)
        self._obs_buffer = deque(maxlen = 4)

    def reset(self):
        for i in range(3):
            self._obs_buffer.append(np.zeros((84, 84)))
        self._obs_buffer.append(self.preprocess(self.env.reset()))
        return self.observe()

    def preprocess(self, frame):
        return mahotas.imresize(mahotas.colors.rgb2grey(frame), (84, 84)) / 255.0
    
    def observe(self):
        return np.stack(self._obs_buffer, axis = -1)

    def step(self, angle, acceleration):
        observation, reward, terminal, info = self.env.step(angle, acceleration)
        self._obs_buffer.append(self.preprocess(observation))
        return self.observe(), reward, terminal, info
