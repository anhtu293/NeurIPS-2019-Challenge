# coding=utf-8
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
import sys
from osim.env import L2M2019Env
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import random
from actor import ActorNetwork
from critic import CriticNetwork
from ou_noise import OUNoise
from utils import Tools
import argparse
import itertools
import time
from replay_buffer import ReplayBuffer

"""
TAU = 0.001
LR = 0.0001
BUFFER_SIZE = 1000000
BATCH_SIZE = 64
DISCOUNT = 0.99
NOISE = 0.5
NOISE_DECAY = 0.99
EPSILON = 1
EPSILON_DECAY = 0.99
"""


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", default="forward", help="direction of falling")
    parser.add_argument("--episodes", default=10000, type=int, help="number of episodes for training")
    parser.add_argument("--TAU", default=0.001, type=float, help="TAU for updating target model")
    parser.add_argument("--lr_actor", default=0.0001, type=float, help="learning rate for actor")
    parser.add_argument("--lr_critic", default=0.001, type=float, help="learning rate for critic")
    parser.add_argument("--buffer_size", default=1000000, type=int, help="buffer size")
    parser.add_argument("--init_buffer_size", default=40, type=int, help="initial size of buffer")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--discount", default=0.99, type=float, help="discount factor")
    parser.add_argument("--noise_decay", default=0.2, type=float, help="noise decay")
    args = parser.parse_args()
    return args


class Actor_Critic:
    def __init__(self, env, args):
        self.env = env
        self.memory_buffer = ReplayBuffer(args.buffer_size)
        self.learning_rate_actor = args.lr_actor
        self.learning_rate_critic = args.lr_critic
        self.tau = args.TAU
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.states_ph = tf.placeholder(tf.float32, shape=((None,) + self.env.observation_space.shape))
        self.actions_ph = tf.placeholder(tf.float32, shape=((None,) + self.env.action_space.shape))
        self.is_training_ph = tf.placeholder_with_default(True, shape=None)
        self.Actor = ActorNetwork(env=self.env, states=self.states_ph, LR=self.learning_rate_actor, TAU=self.tau,
                                  discount=self.discount, scope="actor_main", batch_size=self.batch_size,
                                  is_training=False)
        self.Critic = CriticNetwork(env=self.env, states=self.states_ph, actions=self.actions_ph,
                                    LR=self.learning_rate_critic,
                                    TAU=self.tau, discount=self.discount, scope="critic_main",
                                    batch_size=self.batch_size,
                                    is_training=self.is_training_ph)
        self.Actor_target = ActorNetwork(env=self.env, states=self.states_ph, LR=self.learning_rate_actor, TAU=self.tau,
                                         discount=self.discount, scope="actor_target", batch_size=self.batch_size,
                                         is_training=self.is_training_ph)
        self.Critic_target = CriticNetwork(env=self.env, states=self.states_ph, actions=self.actions_ph,
                                           LR=self.learning_rate_critic,
                                           TAU=self.tau, discount=self.discount, scope="critic_target",
                                           batch_size=self.batch_size, is_training=self.is_training_ph)


class Trainer():
    def __init__(self, model, env, sess, args):
        self.model = model
        self.sess = sess
        directions = {"left": np.pi / 2, "right": -np.pi / 2, "forward": 1}
        self.direction = args.direction
        self.env = env
        self.num_episodes = args.episodes
        self.episode_start = 0
        self.noise = OUNoise(mu=np.zeros(self.env.action_space.shape))
        self.noise_decay = args.noise_decay
        self.count_exp_replay = 0
        self.train_iteration = 0
        self.tau = args.TAU
        self.tools = Tools()
        self.saver = tf.train.Saver()
        self.target_Q_ph = tf.placeholder(tf.float32, shape=(None, 1))
        self.actions_grads_ph = tf.placeholder(tf.float32, shape=((None,) + self.env.action_space.shape))

        # self.sess.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver()

    def load_model(self):
        # self.saver = tf.train.import_meta_graph('./checkpoints/left_1999_model.ckpt.meta')
        self.saver.restore(self.sess, "./checkpoints/left_1999_model.ckpt")
        print("Load successful ! ")

    def visualisation(self):
        for i_episode in range(10):
            state = self.env.reset(obs_as_dict=False)
            state = np.asarray(state)
            self.noise.reset()
            one_episode_score = 0

            for i_step in itertools.count():
                action = self.sess.run(self.model.Actor.output, feed_dict={
                    self.model.states_ph: np.expand_dims(state, 0)
                })[0]

                # execute action action_with_noise and observe reward r_t and s_t+1
                next_state, reward, done, _ = self.env.step(action, obs_as_dict=False)

                reward = self.tools.get_reward(self.direction, self.env.get_state_desc())

                next_state = np.asarray(next_state)
                state = np.copy(next_state)

                print("Time step {} test {} =>>>>>>> reward {} :".format(i_step, i_episode, reward))
                one_episode_score += reward

                if done or i_step == 50000:
                    print("Episode {} =>>>>> Score {}".format(i_episode + 1, one_episode_score))
                    break


if __name__ == '__main__':
    args = arg_parser()
    env = L2M2019Env(visualize=True)
    # Create session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # tf.reset_default_graph()
    model = Actor_Critic(env, args)
    trainer = Trainer(model, env, sess, args)
    trainer.load_model()
    trainer.visualisation()
    print("======= Training Completed =======\n")