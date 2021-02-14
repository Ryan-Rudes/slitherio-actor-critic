from random import sample, choices
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, maxlen):
        self._queue = deque(maxlen = maxlen)
        self.maxlen = maxlen

    def store(self, state, length, action, reward, next_state, next_length):
        self._queue.append((state, length, action, reward, next_state, next_length))

    def sample(self, n):
        if len(self._queue) >= n:
            minibatch = sample(self._queue, k = n)
        else:
            minibatch = choices(self._queue, k = n)

        states = np.array([transition[0] for transition in minibatch])
        lengths = np.array([transition[1] for transition in minibatch])
        actions = np.array([transition[2] for transition in minibatch])
        rewards = np.array([transition[3] for transition in minibatch])
        next_states = np.array([transition[4] for transition in minibatch])
        next_lengths = np.array([transition[5] for transition in minibatch])

        return states, lengths, actions, rewards, next_states, next_lengths