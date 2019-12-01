#**********************************************************************
#This is class Actor to build ddpg actor-critic
#and actor-critic ensemble.
#
#Prorieties :
#    - env : environment used for training
#    - actor_model : actor network
#    - actor_target : actor target network
#    - critic_gradients_output : tensorflow placeholder to pass values 
#    of output of critic model, used for calculating gradients of critic
#    dQ/da
#    - critic_gradients_action : tensorflow placeholder to pass values
#    of input action of critic model, used for calculating gradients of
#    critic dQ/da
#    - critic_gradients : gradients of critic, dQ/da 
#    - actor_gradients : gradients of actor dmu/dtheta
#
#Methods:
#1) _buid_actor_model : build model for actor network and actor target network
#2) actor_target_update : update weights for actor_target
#3) actor_train : train actor network
#**********************************************************************

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

class ActorNetwork:
    def __init__(self, env, states, LR = 0.0001, TAU = 0.125, discount = 0.99, batch_size = 32, scope = "actor",
                 is_training=False):
        self.env = env
        self.learning_rate = LR
        self.TAU = TAU
        self.discount = discount
        self.scope = scope
        self.action_sup = self.env.action_space.high
        self.action_inf = self.env.action_space.low
        self.batch_size = batch_size
        self.is_training = is_training

        #build network
        self.states = states
        with tf.variable_scope(self.scope):
            self.input_state = tf.layers.batch_normalization(self.states, momentum=0.9,fused=True,
                                                             training=self.is_training)
            with tf.variable_scope('dense1'):
                self.dense1_mlp = tf.layers.dense(self.input_state, 400,
                                                  kernel_initializer=tf.random_uniform_initializer(
                                                      (-1 / tf.sqrt(tf.to_float(self.env.observation_space.shape[0]))),
                                                      1 / tf.sqrt(tf.to_float(self.env.observation_space.shape[0]))),
                                                  bias_initializer=tf.random_uniform_initializer(
                                                      (-1 / tf.sqrt(tf.to_float(self.env.observation_space.shape[0]))),
                                                      1 / tf.sqrt(tf.to_float(self.env.observation_space.shape[0])))
                                                  )

                self.dense1_batchnorm = tf.layers.batch_normalization(self.dense1_mlp,fused=True, training=self.is_training)
                self.dense1 = tf.nn.relu(self.dense1_batchnorm)
            with tf.variable_scope('denes2'):
                self.dense2_mlp = tf.layers.dense(self.dense1, 300,
                                                  kernel_initializer=tf.random_uniform_initializer(
                                                      (-1 / tf.sqrt(tf.to_float(400))), 1 / tf.sqrt(tf.to_float(400))),
                                                  bias_initializer=tf.random_uniform_initializer(
                                                      (-1 / tf.sqrt(tf.to_float(400))), 1 / tf.sqrt(tf.to_float(400))))
                self.dense2_batchnorm = tf.layers.batch_normalization(self.dense2_mlp,fused=True, training=self.is_training)
                self.dense2 = tf.nn.relu(self.dense2_batchnorm)
            with tf.variable_scope('output'):
                self.output_mlp = tf.layers.dense(self.dense2, self.env.action_space.shape[0],
                                kernel_initializer=tf.random_uniform_initializer(-1*0.003, 0.003),
                                bias_initializer=tf.random_uniform_initializer(-0.003, 0.003))
                self.output_tanh = tf.nn.tanh(self.output_mlp)

                self.output = tf.add(tf.multiply(tf.to_float(1/2), self.output_tanh), tf.to_float(1/2))

                self.network_params = tf.trainable_variables(scope = self.scope)

    def train_step(self, action_grads):
        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.grads = tf.gradients(self.output, self.network_params, -action_grads)
                self.grads_scaled = list(map(lambda x: tf.divide(x, self.batch_size), self.grads))

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.scope)
                with tf.control_dependencies(update_ops):
                    train_step = self.optimizer.apply_gradients(zip(self.grads_scaled, self.network_params))

                return train_step