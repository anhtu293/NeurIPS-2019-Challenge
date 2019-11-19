#**********************************************************************
#This is class Critic to build ddpg actor-critic
#and actor-critic ensemble.
#
#Prorieties :
#    - env : environment used for training
#    - critic_model : critic network
#    - critic_target : critic target network
#
#Methods:
#1) _buid_critic_model : build model for critic network and critic 
#target network
#2) critic_target_update : update weights for critic_target
#3) critic_train : train critic network
#**********************************************************************

from osim.env import L2M2019Env
import numpy as np
from tensorflow.keras.layers import Dense, Input, Add, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from matplotlib import pyplot as plt
from collections import deque
import tensorflow as tf
import tensorflow.keras.backend as K
import random
import replay_buffer

def nth_root(num, n):
    return(n**(1/num))

class CriticNetwork:
    def __init__(self, env, states, actions, LR = 0.0001, TAU = 0.125, discount = 0.99, batch_size = 32,
                 scope = "critic", is_training=False):
        self.env = env
        self.env = env
        self.learning_rate = LR
        self.tau = TAU
        self.discount = discount
        self.discount_decay = 0.99
        self.batch_size = batch_size
        self.scope = scope
        self.is_training = is_training

        self.states = states
        self.actions = actions
        with tf.variable_scope(self.scope):
            self.input_state = tf.layers.batch_normalization(self.states, training=self.is_training)
            with tf.variable_scope("dense1"):
                self.dense1_mlp = tf.layers.dense(self.input_state, 400,
                                                  kernel_initializer=tf.random_uniform_initializer(
                                                      (-1 / tf.sqrt(tf.to_float(self.env.observation_space.shape[0]))),
                                                      1 / tf.sqrt(tf.to_float(self.env.observation_space.shape[0]))),
                                                  bias_initializer=tf.random_uniform_initializer(
                                                      (-1 / tf.sqrt(tf.to_float(self.env.observation_space.shape[0]))),
                                                      1 / tf.sqrt(tf.to_float(self.env.observation_space.shape[0])))
                                                  )
                self.dense1_batchnorm = tf.layers.batch_normalization(self.dense1_mlp, training=self.is_training)
                self.dense1 = tf.nn.relu(self.dense1_batchnorm)
            with tf.variable_scope("dense2"):
                self.dense2a = tf.layers.dense(self.dense1, 300,
                                           kernel_initializer=tf.random_uniform_initializer(
                                               (-1 / tf.sqrt(tf.to_float(400 + self.env.action_space.shape[0]))),
                                               1 / tf.sqrt(tf.to_float(400 + self.env.action_space.shape[0]))),
                                           bias_initializer=tf.random_uniform_initializer(
                                               (-1 / tf.sqrt(tf.to_float(400 + self.env.action_space.shape[0]))),
                                               1 / tf.sqrt(tf.to_float(400 + self.env.action_space.shape[0])))
                                               )
                self.dense2b = tf.layers.dense(self.actions, 300,
                                               kernel_initializer=tf.random_uniform_initializer(
                                                   (-1 / tf.sqrt(tf.to_float(400 + self.env.action_space.shape[0]))),
                                                   1 / tf.sqrt(tf.to_float(400 + self.env.action_space.shape[0]))),
                                               bias_initializer=tf.random_uniform_initializer(
                                                   (-1 / tf.sqrt(tf.to_float(400 + self.env.action_space.shape[0]))),
                                                   1 / tf.sqrt(tf.to_float(400 + self.env.action_space.shape[0])))
                                               )
                self.dense2 = tf.nn.relu(self.dense2a + self.dense2b)
            with tf.variable_scope("output"):
                self.output = tf.layers.dense(self.dense2, 1,
                                              kernel_initializer=tf.random_uniform_initializer(-1 * 0.003, 0.003),
                                              bias_initializer=tf.random_uniform_initializer(-0.003, 0.003))

            self.network_params = tf.trainable_variables(scope = self.scope)

            self.action_grads = tf.gradients(self.output, self.actions)

    def train_step(self, target_Q):
        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                l2_lambda = 0
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.loss = tf.losses.mean_squared_error(target_Q, self.output)
                self.l2_reg_loss = tf.add_n(
                    [tf.nn.l2_loss(v) for v in self.network_params if 'kernel' in v.name]) * l2_lambda
                self.total_loss = self.loss + self.l2_reg_loss

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.scope)
                with tf.control_dependencies(update_ops):
                    train_step = self.optimizer.minimize(self.total_loss, var_list=self.network_params)

                return train_step

class Critic:
    def __init__(self, env, sess, LR = 0.0001, TAU = 0.125, discount = 0.99, batch_size = 32):
        self.env = env
        self.sess = sess
        self.learning_rate = LR
        self.tau = TAU
        self.discount = discount
        self.discount_decay = 0.99
        self.batch_size = batch_size
        print("Summary critic network")
        self.critic_model, self.action, self.state, self.output = self._build_critic_model()
        self.critic_target, self.target_action, self.target_state, self.target_output = self._build_critic_model()

        #self.critic_model_output = self.critic_model.outputs
        self.critic_gradients = tf.gradients(self.output, self.action)

        self.sess.run(tf.global_variables_initializer())
    
    def _build_critic_model(self):
        state_input = Input(shape = [self.env.observation_space.shape[0]])
        state_1 = Dense(200, activation = 'relu')(state_input)
        state_2 = Dense(100, activation = 'relu')(state_1)

        action_input = Input(shape = (self.env.action_space.shape[0]))
        action_1 = Dense(100, activation = 'relu')(action_input)

        merge = Add()([state_2, action_1])
        merge_1 = Dense(50, activation = 'relu')(merge)
        output = Dense(1, activation = 'linear')(merge_1)

        model = Model(inputs = [state_input, action_input], outputs = output)
        model.compile(optimizer = Adam(self.learning_rate), loss = 'mse')
        model.summary()

        return model, action_input, state_input, output
    
    def gradients(self, states, actions):

        return self.sess.run(self.critic_gradients, feed_dict = {
            self.state : states,
            self.action : actions
        })[0]

    def critic_target_update(self):
        critic_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau*critic_weights[i] + (1 - self.tau)*critic_target_weights[i]
        self.critic_target.set_weights(critic_target_weights)
    
    def critic_train(self, actor_target, samples):
        batch_state, batch_action, batch_reward, batch_ns, batch_info = samples
        batch_ns = np.array(batch_ns)
        batch_state = np.array(batch_state)
        batch_reward = np.array(batch_reward)
        batch_info = np.array(batch_info)

        self.discount *= self.discount_decay

        next_actions = actor_target.predict(batch_ns)
        target_Q = self.critic_target.predict([batch_ns, next_actions])
        target_Q = target_Q.reshape([1, target_Q.shape[0]])[0]
        for i in range(len(batch_ns)):
            if not batch_info[i] :
                batch_reward[i] += self.discount*target_Q[i]

        history = self.critic_model.fit([batch_state, batch_action], batch_reward, batch_size = 32, verbose = 0)
        return(history.history["loss"][0])

    def save(self, prefixe):
        name_file = prefixe + "_critic_target.h5"
        self.critic_target.save_weights(name_file)
        name_file = prefixe + "_critic_model.h5"
        self.critic_model.save_weights(name_file)
        print("Saving critic network completed ! \n")
    
