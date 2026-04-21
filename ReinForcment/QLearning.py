import numpy as np
import random

class QAgent:
    def __init__(self, size, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.q = np.zeros((size, size, 4))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def act(self, state):
        x, y = state

        if random.random() < self.epsilon:
            return random.randint(0, 3)

        q_vals = self.q[x, y]
        max_q = np.max(q_vals)
        actions = np.where(q_vals == max_q)[0]

        return random.choice(actions)

    def learn(self, state, action, reward, next_state, done):
        x, y = state
        nx, ny = next_state

        target = reward if done else reward + self.gamma * np.max(self.q[nx, ny])

        self.q[x, y, action] += self.alpha * (target - self.q[x, y, action])