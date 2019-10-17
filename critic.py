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
    def __init__(self, env, sess, LR = 0.001, TAU = 0.125, discount = 0.99, batch_size = 32):
        self.env = env
        self.sess = sess
        self.learning_rate = LR
        self.tau = TAU
        self.discount = discount
        self.batch_size = batch_size
        self.critic_model, self.action, self.state = self._build_critic_model()
        self.critic_target, self.target_action, self.target_state = self._build_critic_model()

        self.critic_model_output = self.critic_model.output
        self.critic_gradients_output = tf.placeholder(tf.float32, [None, 1])
        self.critic_gradients = tf.gradients(self.critic_gradients_output, self.action)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    def _build_critic_model(self):
        state_input = Input(shape = (None,self.env.observation_space.shape[0]))
        state_1 = Dense(200, activation = 'relu')(state_input)
        state_2 = Dense(100, activation = 'relu')(state_1)

        action_input = Input(shape = (None,self.env.action_space.shape[0]))
        action_1 = Dense(100, activation = 'relu')(action_input)

        merge = Add()([state_2, action_1])
        merge_1 = Dense(50, activation = 'relu')(merge)
        output = Dense(1, activation = 'linear')(merge_1)

        model = Model(inputs = [state_input, action_input], outputs = output)
        model.compile(optimizer = Adam(self.learning_rate), loss = 'mse')
        model.summary()

        return model, action_input, state_input
    
    def gradients(self, states, actions):
        self.sess.run(self.gradients, feed = {
            self.state : states,
            self.action : actions
        })

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
        batch_reward = np.array(batch_state)
        batch_info = np.array(batch_info)
        target_actions = actor_target.predict(batch_ns)
        
        target_predicts = self.critic_target.predict([batch_ns, target_actions])
        target_predicts.reshape([1, target_predicts.shape[0]])[0]
        
        for i in range(len(batch_ns)):
            #new_state = [batch_ns[i]]
            if not batch_info :
                batch_reward[i] += self.discount*target_predicts[i]
        
        history = self.critic_model.train_on_batch([batch_state, batch_action], batch_reward)
        return(history)

    def save(self, prefixe):
        name_file = prefixe + "_critic_target.h5"
        self.critic_target.save_weights(name_file)
        name_file = prefixe + "_critic_model.h5"
        self.critic_model.save_weights(name_file)
        print("Saving critic network completed ! \n")
    