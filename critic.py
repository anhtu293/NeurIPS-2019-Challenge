"""
**********************************************************************
This is class Critic to build ddpg actor-critic
and actor-critic ensemble.

Prorieties :
    - env : environment used for training
    - critic_model : critic network
    - critic_target : critic target network

Methods:
1) _buid_critic_model : build model for critic network and critic 
target network
2) critic_target_update : update weights for critic_target
3) critic_train : train critic network


**********************************************************************
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
import replay_buffer


class Critic:
    def __init__(self, env, LR = 0.001, TAU = 0.125, discount = 0.99, batch_size = 32):
        self.env = env
        self.critic_model = self._build_critic_model()
        self.critic_target = self._build_critic_model()
        self.learning_rate = LR
        self.tau = TAU
        self.discount = discount
        self.batch_size = batch_size
    
    def _build_critic_model(self):
        state_input = Input(shape = self.env.observation_space.shape[0])
        state_1 = Dense(200, activation = 'relu')(state_input)
        state_2 = Dense(100, activation = 'relu')(state_1)

        action_input = Input(shape = self.env.action_space.shape[0])
        action_1 = Dense(100, activation = 'relu')(action_input)

        merge = Add()([state_2, action_1])
        merge_1 = Dense(50, activation = 'relu')(merge)
        output = Dense(1, activation = 'relu')(merge_1)

        model = Model(input = [state_input, action_input], output = output)
        model.compile(optimizer = Adam(self.learning_rate), loss = 'mse')
        
        return model
    
    def critic_target_update(self):
        critic_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau*critic_weights[i] + (1 - self.tau)*critic_target_weights[i]
        self.critic_target.set_weights(critic_target_weights)
    
    def critic_train(self, actor_target, samples):
        batch_state, batch_action, batch_reward, batch_ns, batch_info = samples
        for i in range(len(batch_ns)):
            target_action = actor_target.predict(batch_ns[i])
            batch_reward[i] += self.discount*self.critic_target([batch_ns[i], target_action])[0][0]
        self.critic_model.fit([batch_state, batch_action], batch_reward, batch_size = self.batch_size, verbose = 0)
    
