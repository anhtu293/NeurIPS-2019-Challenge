import numpy as np
from collections import deque
import random
import pickle
import gym
#BUFFER_SIZE = 1000000

class Replay_Buffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque()
        self.buffer_size = buffer_size
        self.count = 0
        self.batch_size = batch_size

    def memorize(self, transition):
        if self.count == self.buffer_size:
            self.buffer.popleft()
            self.buffer.append(transition)
        else:
            self.buffer.append(transition)
            self.count += 1

    def size(self):
        return self.count

    def sample_batch(self):
        batch = []
        batch = random.sample(self.buffer, self.batch_size)

        batch_state = np.array([x[0] for x in batch])
        batch_action = np.array([x[1] for x in batch])
        batch_reward = np.array([x[2] for x in batch])
        batch_nextstate = np.array([x[3] for x in batch])
        batch_info = np.array([x[4] for x in batch])
        return batch_state, batch_action, batch_reward, batch_nextstate, batch_info
    
    def save(self, name = "log/memory_buffer.pickle"):
        with open(name, 'wb') as f:
            pickle.dump(self.buffer, f)
    
    def load(self, name = "log/memory_buffer.pickle"):
        with open(name, 'rb') as f:
            self.buffer = pickle.load(f)

if __name__== "__main__":
    env = gym.make("Pendulum-v0")
    buffer = Replay_Buffer(1000, 3)
    for i in range(10):
        state = env.reset()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        buffer.memorize([state, action, reward, next_state, done])
    print(buffer.sample_batch())
    print(buffer.sample_batch())