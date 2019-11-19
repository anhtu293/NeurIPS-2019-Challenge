# *********************************************************
# This is the class Actor_Critic for algorithmddpg actor_critic.
# In this algorithm we use noise decay for exploration of agent.
# Noise is created by Gaussian distribution.
#
# Proprieties :
#    - env : environment used for training
#    - sess: tensorflow session
#    - memory_buffer : memory buffer for experience replay
#    - Actor : actor network
#    - Critic : critic network
#
# Methods :
# 1) update_target : update the weights of target networks
#
# 2) train : train actor network and critic network
# **********************************************************

import sys

sys.path.append("../")
import gym
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
from actor import Actor, ActorNetwork
from critic import Critic, CriticNetwork
from ou_noise import OUNoise
import utils
import argparse
import itertools
import time
from replay_buffer2 import ReplayMemory
TAU = 0.001
LR = 0.0001
BUFFER_SIZE = 1000000
BATCH_SIZE = 64
DISCOUNT = 0.99
NOISE = 0.5
NOISE_DECAY = 0.99
EPSILON = 1
EPSILON_DECAY = 0.99


def nth_root(num, n):
    return (n ** (1 / num))


class Actor_Critic:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess
        self.memory_buffer = ReplayMemory(BUFFER_SIZE, 20000, BATCH_SIZE, env.observation_space.shape, env.action_space.shape)
        self.learning_rate = LR
        self.tau = TAU
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.discount = 0.99
        self.states_ph = tf.placeholder(tf.float32, shape=((None,) + self.env.observation_space.shape))
        self.actions_ph = tf.placeholder(tf.float32, shape=((None,) + self.env.action_space.shape))
        self.is_training_ph = tf.placeholder_with_default(True, shape=None)
        self.Actor = ActorNetwork(env=self.env, states=self.states_ph, LR = self.learning_rate, TAU = self.tau,
                                  discount=self.discount, scope="actor_main", batch_size=self.batch_size, is_training=self.is_training_ph)
        self.Critic = CriticNetwork(env=self.env, states=self.states_ph, actions=self.actions_ph, LR=self.learning_rate,
                                    TAU=self.tau, discount=self.discount, scope="critic_main", batch_size=self.batch_size, is_training=self.is_training_ph)
        self.Actor_target = ActorNetwork(env=self.env, states = self.states_ph, LR = 0.001, TAU = self.tau,
                                         discount=self.discount, scope="actor_target", batch_size=self.batch_size, is_training=self.is_training_ph)
        self.Critic_target = CriticNetwork(env=self.env, states=self.states_ph, actions=self.actions_ph, LR=self.learning_rate,
                                           TAU=self.tau, discount=self.discount, scope="critic_target", batch_size=self.batch_size, is_training=self.is_training_ph)

    def update_target_network(self,network_params, target_network_params, tau = 1):
        op_holder = []
        for from_var, to_var in zip(network_params, target_network_params):
            op_holder.append(to_var.assign((tf.multiply(from_var, tau) + tf.multiply(to_var, 1. - tau))))

        return op_holder

    def save(self, prefixe):
        self.Actor.save(prefixe)
        self.Critic.save(prefixe)
        self.memory_buffer.save()


class Trainer():
    def __init__(self, model, env, sess, num_episodes, direction):
        self.model = model
        self.sess = sess
        self.direction = direction
        self.env = env
        self.num_episodes = num_episodes
        self.episode_start = 0
        self.noise = OUNoise(mu=np.zeros(self.env.action_space.shape))
        self.noise_decay = 0.2
        self.epsilon = EPSILON
        self.epsilon_decay = nth_root(self.num_episodes, 0.001 / self.epsilon)
        self.count_exp_replay = 0
        self.tau = TAU
        self.target_Q_ph = tf.placeholder(tf.float32, shape=(None,1))
        self.actions_grads_ph = tf.placeholder(tf.float32, shape=((None,) + self.env.action_space.shape))

        #train operation
        self.actor_train_ops = self.model.Actor.train_step(self.actions_grads_ph)
        self.critic_train_ops = self.model.Critic.train_step(self.target_Q_ph)

        #update operation
        self.update_critic_target = self.model.update_target_network(self.model.Critic.network_params,
                                                                     self.model.Critic_target.network_params, self.tau)
        self.update_actor_target = self.model.update_target_network(self.model.Actor.network_params,
                                                                    self.model.Actor_target.network_params, self.tau)
        sess.run(tf.initialize_all_variables())
        #for testing only
        self.sess.run(self.model.update_target_network(self.model.Critic.network_params,self.model.Critic_target.network_params))
        self.sess.run(self.model.update_target_network(self.model.Actor.network_params, self.model.Actor_target.network_params))

        # reward summary for tensorboard
        self.tf_reward = tf.Variable(0.0, trainable=False, name='reward_summary')
        self.tf_reward_summary = tf.summary.scalar("Reward by episode", self.tf_reward)

        # time
        self.tf_time = tf.Variable(0.0, trainable=False, name='Time_per_episode')
        self.tf_time_summary = tf.summary.scalar("Time per episode", self.tf_time)

        # writer
        self.writer = tf.summary.FileWriter('./graphs', self.sess.graph)

    def tryLoadWeights(self):
        print("Load weights \n")
        try:
            with open("./log/data.txt", 'r') as f:
                data = f.read()
            num_episodes, model_name_prefix, noise, epsilon = data.split(" ")
            self.episode_start = num_episodes
            self.num_episodes += num_episodes
            self.noise = noise
            self.epsilon = epsilon
            print("Log loaded !\n")
            self.Actor.actor_model.load_weights(model_name_prefix + "_actor_model.h5")
            self.Actor.actor_target.load_weights(model_name_prefix + "_actor_target.h5")
            self.Critic.critic_model.load_weights(model_name_prefix + "_critic_model.h5")
            self.Critic.critic_model.load_weights(model_name_prefix + "_critic_target.h5")
            print("Weights load successfully ! \n")
            self.memory_buffer.load()
            print("Memory buffer load succesfully ! \n")
            return (0)
        except:
            # if self.episode_start == 0:
            #    return(False)
            print("New training \n")
            return (1)

    def play_to_init_buffer(self):
        """
        for i_episode in range(1):
            # reset env
            state = env.reset()

            action_with_noise = np.zeros([1, self.model.env.action_space.shape[0]])

            for i_step in itertools.count():
                # state = state.reshape(1,3)
                action_original = self.model.env.action_space.sample()
                action_with_noise = action_original

                # execute action action_with_noise and observe reward r_t and s_t+1
                state, reward, done, _ = self.model.env.step(action_with_noise)

                # reward = -reward

                self.model.memory_buffer.add(action_with_noise, reward, state, done)

                if done:
                    break
                #else:
                #    state = next_state
        """
        self.env.reset()
        for random_step in range(1, 50000 + 1):
            self.env.render()
            action = env.action_space.sample()
            state, reward, terminal, _ = self.env.step(action)
            self.model.memory_buffer.add(action, reward, state, terminal)

            if terminal:
                self.env.reset()

    def DDPG(self, model_name_prefix):
        scores = []
        for i_episode in range(self.episode_start, self.num_episodes):
            start = time.time()
            one_episode_score = 0

            # reset env
            state = self.env.reset()

            self.noise.reset()

            for i_step in itertools.count():
                self.env.render()
                action = self.sess.run(self.model.Actor.output, feed_dict={
                    self.model.states_ph : np.expand_dims(state,0),
                    self.model.is_training_ph : False
                })[0]
                action += self.noise() * self.noise_decay
                # execute action action_with_noise and observe reward r_t and s_t+1
                state, reward, done, _ = self.env.step(action)
                self.model.memory_buffer.add(action, reward, state, done)

                one_episode_score += reward

                #self.experience_replay()
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.model.memory_buffer.getMinibatch()

                future_action = self.sess.run(self.model.Actor_target.output, feed_dict={
                    self.model.states_ph: batch_next_state
                })
                future_Q = self.sess.run(self.model.Critic_target.output, feed_dict={
                    self.model.states_ph: batch_next_state,
                    self.model.actions_ph: future_action
                })[:, 0]

                future_Q[batch_done] = 0
                targets = batch_reward + (future_Q * DISCOUNT)
                # train Critic
                self.sess.run(self.critic_train_ops, feed_dict={
                    self.model.states_ph: batch_state,
                    self.model.actions_ph: batch_action,
                    self.target_Q_ph: np.expand_dims(targets, 1)
                })

                # train Actor
                actor_actions = self.sess.run(self.model.Actor.output, feed_dict={
                    self.model.states_ph: batch_state
                })
                action_grads = self.sess.run(self.model.Critic.action_grads, feed_dict={
                    self.model.states_ph: batch_state,
                    self.model.actions_ph: actor_actions
                })
                self.sess.run(self.actor_train_ops, feed_dict={
                    self.model.states_ph: batch_state,
                    self.actions_grads_ph: action_grads[0]
                })
                # update target
                self.sess.run(self.update_critic_target)
                self.sess.run(self.update_actor_target)

                if done or i_step == 50000:
                    end = time.time()
                    print("Episode {} =>>>>> Score {}".format(i_episode, one_episode_score))
                    scores.append(one_episode_score)
                    # write reward for tensorboard
                    summary = self.sess.run(self.tf_reward_summary, feed_dict={
                        self.tf_reward: one_episode_score
                    })
                    # add summary to writer
                    self.writer.add_summary(summary, i_episode)
                    # timer
                    summary = self.sess.run(self.tf_time_summary, feed_dict={
                        self.tf_time: end - start
                    })
                    self.writer.add_summary(summary, i_episode)
                    break
        # save model
        model.save("./models/{}_{}".format(args.direction, args.episodes))
        print("Models saved successfully ! \n")

    def experience_replay(self):
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.model.memory_buffer.getMinibatch()

        future_action = self.sess.run(self.model.Actor_target.output, feed_dict={
            self.model.states_ph : batch_next_state
        })
        future_Q = self.sess.run(self.model.Critic_target.output, feed_dict={
            self.model.states_ph : batch_next_state,
            self.model.actions_ph : future_action
        })[:,0]

        future_Q[batch_done] = 0
        targets = batch_reward + (future_Q * DISCOUNT)
        #train Critic
        self.sess.run(self.critic_train_ops, feed_dict={
            self.model.states_ph : batch_state,
            self.model.actions_ph : batch_action,
            self.target_Q_ph : np.expand_dims(targets,1)
        })

        #train Actor
        actor_actions = self.sess.run(self.model.Actor.output, feed_dict={
            self.model.states_ph : batch_state
        })
        action_grads = self.sess.run(self.model.Critic.action_grads, feed_dict={
            self.model.states_ph : batch_state,
            self.model.actions_ph: actor_actions
        })
        self.sess.run(self.actor_train_ops, feed_dict={
            self.model.states_ph : batch_state,
            self.actions_grads_ph : action_grads[0]
        })
        #update target
        self.sess.run(self.update_critic_target)
        self.sess.run(self.update_actor_target)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", default="forward", help="direction of falling")
    parser.add_argument("--out", default="front", help="prefix of output file")
    parser.add_argument("--episodes", default=10000, help="number of episodes for training")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()
    env = gym.make("Pendulum-v0")
    # Create session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model = Actor_Critic(env, sess)
    print("======= Start Training =======\n")

    trainer = Trainer(model, env, sess, args.episodes, args.direction)
    if trainer.tryLoadWeights() == 1:
        print("Play to initiate buffer !")
        trainer.play_to_init_buffer()

    trainer.DDPG(model_name_prefix="./models/" + args.direction)
    print("======= Training Completed =======\n")
