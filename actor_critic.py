"""
*********************************************************
This is the class Actor_Critic for algorithm
ddpg actor_critic. 

Proprieties : 
    - env : environment used for training
    - sess: tensorflow session
    - memory_buffer : memory buffer for experience replay
    - Actor : actor network 
    - Critic : critic network

Methods :
1) update_target : update the weights of 
target networks

2) train : train actor network and critic 
network
**********************************************************
"""

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
from replay_buffer import Replay_Buffer
from actor import Actor
from critic import Critic

TAU = 0.125
LR = 0.001
BUFFER_SIZE = 10000
BATCH_SIZE = 32
DISCOUNT = 0.99

class Actor_Critic():
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess
        self.memory_buffer = Replay_Buffer(BUFFER_SIZE, BATCH_SIZE)
        self.learning_rate = LR
        self.tau = TAU
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.discount = 0.99
        self.Actor = Actor(env, sess, self.learning_rate, self.tau, self.discount)
        self.Critic = Critic(env, self.learning_rate, self.tau, self.discount)

    def update_target(self):
        self.Actor.actor_target_update()
        self.Critic.critic_target_update()
    
    def train(self):
        samples = self.memory_buffer.sample_batch()
        self.Actor.actor_train(self.Critic.critic_model, samples)
        self.Critic.critic_train(self.Actor.actor_target, samples)

