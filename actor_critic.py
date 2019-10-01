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

TAU = 0.125
LR = 0.001
BUFFER_SIZE = 10000
BATCH_SIZE = 32

class Actor_Critic():
    def __init__(self, env, sess):
        self.sess = sess
        self.env = env
        self.actor_model = self.build_actor_model()
        self.critic_model = self.build_critic_model()
        self.actor_target = self.build_actor_model()
        self.critic_target = self.build_critic_model()
        self.memory_buffer = Replay_Buffer()
        self.tau = TAU
        self.learning_rate = LR
        self.discount = 0.99

        K.set_session(sess)
                    #UPDATE ACTOR NETWORK
        self.critic_gradients_state = tf.placeholder(tf.float32, [None, self.env.observation_space.shape[0]])
        self.critic_gradients_action = tf.placeholder(tf.float32, [None, self.env.action_space.shape[0]])
        self.critic_gradients = tf.gradients(self.critic_gradients_state, self.critic_gradients_action)
        self.actor_gradients = tf.gradients(self.actor_model.output, self.actor_model.trainable_weights, -self.critic_gradients)
        grads = zip(self.actor_gradients, self.actor_model.trainable_weights)
        self.actor_update = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

    def build_actor_model(self):
        model = Sequential()
        model.add(Dense(200, activation = 'relu'))
        model.add(Dense(100, activation = 'relu'))
        model.add(Dense(self.env.action_space.shape[0], activation = "linear"))
        model.compile(optimizer = Adam(self.learning_rate), loss = "mse")
        return model

    def build_critic_model(self):
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

    def _actor_target_update(self):
        actor_weights = self.actor_model.get_weights()
        actor_target_weights = self.actor_target.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau*actor_weights[i] + (1 - self.tau)*actor_target_weights[i]
        self.actor_target.set_weights(actor_target_weights) 

    def _critic_target_update(self):
        critic_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau*critic_weights[i] + (1 - self.tau)*critic_target_weights[i]
        self.critic_target.set_weights(critic_target_weights)
    
    def update_target(self):
        self._actor_target_update()
        self._critic_target_update()
    

    def _critic_train(self, samples):
        batch_state, batch_action, batch_reward, batch_ns, batch_info = samples
        for i in range(len(batch_ns)):
            target_action = self.actor_target.predict(batch_ns[i])
            batch_reward[i] += self.discount*self.critic_target([batch_ns[i], target_action])[0][0]
        self.critic_model.fit([batch_state, batch_action], batch_reward, batch_size = self.memory_buffer.batch_size, verbose = 0)
    
    def _actor_train(self, samples):
        batch_state, batch_action, batch_reward, batch_ns, batch_info = samples
        predicted_action = self.actor_model.predict(batch_state)
        gradients = self.sess.run(self.critic_gradients, feed_dict = {
            self.critic_gradients_state : batch_state,
            self.critic_gradients_action : batch_action
        })

        self.sess.run(self.actor_update, feed_dict= {
            self.critic_gradients : gradients
        })

    def train(self):
        samples = self.memory_buffer.sample_batch()
        self._critic_train(samples)
        self._actor_train(samples)