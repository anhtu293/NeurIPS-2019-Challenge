"""
**********************************************************************
This is class Actor to build ddpg actor-critic
and actor-critic ensemble.

Prorieties :
    - env : environment used for training
    - actor_model : actor network
    - actor_target : actor target network
    - critic_gradients_output : tensorflow placeholder to pass values 
    of output of critic model, used for calculating gradients of critic
    dQ/da
    - critic_gradients_action : tensorflow placeholder to pass values
    of input action of critic model, used for calculating gradients of
    critic dQ/da
    - critic_gradients : gradients of critic, dQ/da 
    - actor_gradients : gradients of actor dmu/dtheta

Methods:
1) _buid_actor_model : build model for actor network and actor target
network
2) actor_target_update : update weights for actor_target
3) actor_train : train actor network


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


class Actor:
    def __init__(self, env, sess, LR = 0.001, TAU = 0.125, discount = 0.99):
        self.env = env
        self.sess = sess
        self.actor_model = self._build_actor_model()
        self.actor_target = self._build_actor_model()
        self.learning_rate = LR
        self.tau = TAU
        self.discount = discount

        #UPDATE ACTOR NETWORK
        self.critic_gradients_output = tf.placeholder(tf.float32, [None, 1])
        self.critic_gradients_action = tf.placeholder(tf.float32, [None, self.env.action_space.shape[0]])
        self.critic_gradients = tf.gradients(self.critic_gradients_output, self.critic_gradients_action)
        self.actor_gradients = tf.gradients(self.actor_model.output, self.actor_model.trainable_weights, -self.critic_gradients)
        grads = zip(self.actor_gradients, self.actor_model.trainable_weights)
        self.actor_update = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
    
    def _build_actor_model(self):
        model = Sequential()
        model.add(Dense(200, activation = 'relu'))
        model.add(Dense(100, activation = 'relu'))
        model.add(Dense(self.env.action_space.shape[0], activation = "linear"))
        model.compile(optimizer = Adam(self.learning_rate), loss = "mse")
        return model

    def actor_target_update(self):
        actor_weights = self.actor_model.get_weights()
        actor_target_weights = self.actor_target.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau*actor_weights[i] + (1 - self.tau)*actor_target_weights[i]
        self.actor_target.set_weights(actor_target_weights) 

    def actor_train(self, critic_model, samples):
        batch_state, batch_action, batch_reward, batch_ns, batch_info = samples
        #predicted_action = self.actor_model.predict(batch_state)
        critic_model_output = critic_model.output
        gradients = self.sess.run(self.critic_gradients, feed_dict = {
            self.critic_gradients_output : critic_model_output,
            self.critic_gradients_action : batch_action
        })

        self.sess.run(self.actor_update, feed_dict= {
            self.critic_gradients : gradients
        })

