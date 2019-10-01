from osim.env import L2M2019Env
import numpy as np
from tensorflow.keras.layers import Dense, Input, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from matplotlib import pyplot as plt
from collections import deque
import tensorflow as tf
import tensorflow.keras.backend as K
import random
import replay_buffer
import actor_critic


TAU = 0.125
LR = 0.001
BUFFER_SIZE = 10000
BATCH_SIZE = 32


    


if __name__ == '__main__':
    env = L2M2019Env(visualize=True)
    observation = env.reset()
    print(env.action_space.shape)
    print(env.observation_space.shape)