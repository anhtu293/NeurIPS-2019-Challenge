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
from replay_buffer import ReplayMemory
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
        self.memory_buffer = ReplayMemory(args.buffer_size, args.init_buffer_size,
                                          args.batch_size, env.observation_space.shape, env.action_space.shape)
        self.learning_rate_actor = args.lr_actor
        self.learning_rate_critic = args.lr_critic
        self.tau = args.TAU
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.states_ph = tf.placeholder(tf.float32, shape=((None,) + self.env.observation_space.shape))
        self.actions_ph = tf.placeholder(tf.float32, shape=((None,) + self.env.action_space.shape))
        self.is_training_ph = tf.placeholder_with_default(True, shape=None)
        self.Actor = ActorNetwork(env=self.env, states=self.states_ph, LR = self.learning_rate_actor, TAU = self.tau,
                                  discount=self.discount, scope="actor_main", batch_size=self.batch_size,
                                  is_training=self.is_training_ph)
        self.Critic = CriticNetwork(env=self.env, states=self.states_ph, actions=self.actions_ph, LR=self.learning_rate_critic,
                                    TAU=self.tau, discount=self.discount, scope="critic_main", batch_size=self.batch_size,
                                    is_training=self.is_training_ph)
        self.Actor_target = ActorNetwork(env=self.env, states = self.states_ph, LR = self.learning_rate_actor, TAU = self.tau,
                                         discount=self.discount, scope="actor_target", batch_size=self.batch_size,
                                         is_training=self.is_training_ph)
        self.Critic_target = CriticNetwork(env=self.env, states=self.states_ph, actions=self.actions_ph, LR=self.learning_rate_critic,
                                           TAU=self.tau, discount=self.discount, scope="critic_target",
                                           batch_size=self.batch_size, is_training=self.is_training_ph)

    def update_target_network(self,network_params, target_network_params, tau = 1):
        op_holder = []
        for from_var, to_var in zip(network_params, target_network_params):
            op_holder.append(to_var.assign((tf.multiply(from_var, tau) + tf.multiply(to_var, 1. - tau))))

        return op_holder
    """
    def save(self, prefixe):
        self.Actor.save(prefixe)
        self.Critic.save(prefixe)
        self.memory_buffer.save()
    """

class Trainer():
    def __init__(self, model, env, sess, args):
        self.model = model
        self.sess = sess
        directions = {"left" : np.pi/2, "right" : -np.pi/2, "forward" : 1}
        self.direction = args.direction
        self.env = env
        self.num_episodes = args.episodes
        self.episode_start = 0
        self.noise = OUNoise(mu=np.zeros(self.env.action_space.shape))
        self.noise_decay = args.noise_decay
        self.count_exp_replay = 0
        self.tau = args.TAU
        self.tools = Tools()
        self.saver = tf.train.Saver()
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

        #init target networks by copying weights from actor and critic network
        self.sess.run(self.model.update_target_network(self.model.Critic.network_params,self.model.Critic_target.network_params))
        self.sess.run(self.model.update_target_network(self.model.Actor.network_params, self.model.Actor_target.network_params))

        # reward summary for tensorboard
        self.tf_reward = tf.Variable(0.0, trainable=False, name='Reward_per_episode')
        self.tf_reward_summary = tf.summary.scalar("Reward by episode", self.tf_reward)

        # time
        self.tf_time = tf.Variable(0.0, trainable=False, name='Time_per_episode')
        self.tf_time_summary = tf.summary.scalar("Time per episode", self.tf_time)

        # step
        self.tf_step = tf.Variable(0.0, trainable=False, name='Step_per_episode')
        self.tf_step_summary = tf.summary.scalar("Step per episode", self.tf_step)

        # writer
        self.writer = tf.summary.FileWriter('./graphs', self.sess.graph)

    def saveWeights(self, prefixe):
        path = prefixe + "model.ckpt"
        save = self.saver.save(self.sess, path)

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
            """
            self.Actor.actor_model.load_weights(model_name_prefix + "_actor_model.h5")
            self.Actor.actor_target.load_weights(model_name_prefix + "_actor_target.h5")
            self.Critic.critic_model.load_weights(model_name_prefix + "_critic_model.h5")
            self.Critic.critic_model.load_weights(model_name_prefix + "_critic_target.h5")
            """
            self.saver.restore(self.sess, "./model/model.ckpt")
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
        self.env.reset(obs_as_dict=False)
        for random_step in range(1, args.init_buffer_size + 1):
            #self.env.render()
            print("\r Examples : {}/{}".format(random_step, args.init_buffer_size), end="")
            sys.stdout.flush()
            action = self.env.action_space.sample()
            state, reward, terminal, _ = self.env.step(action, obs_as_dict=False)
            reward = 0
            if terminal :
                reward = self.tools.get_reward(self.direction, self.env.get_state_desc())
            state = np.asarray(state)
            self.model.memory_buffer.add(action, reward, state, terminal)

            if terminal:
                self.env.reset()

    def DDPG(self, model_name_prefix):
        scores = []
        for i_episode in range(self.episode_start, self.num_episodes):
            start = time.time()
            one_episode_score = 0

            # write log of training
            name = "./log/training.txt"
            with open(name, 'a') as f:
                f.write("Episode {}/{} \n".format(i_episode + 1, self.num_episodes))
            f.close()

            if (i_episode + 1) % 100 == 0:
                avg = np.mean(np.asarray(scores))
                if (i_episode + 1) % 1000 == 0:
                    #self.model.Actor.save(prefixe=prefixe + "checkpoint_avgScore_{}".format(avg))
                    #self.model.Critic.save(prefixe=prefixe + "checkpoint_avgScore_{}".format(avg))
                    self.saveWeights("./checkpoints/{}_{}_".format(args.direction, i_episode))
                    self.noise_decay *= 0.8
                print(
                    "Episode {}/{} : Average score in 100 latest episodes : {}".format(i_episode + 1, self.num_episodes,
                                                                                       avg))
                scores.clear()

            # reset env
            state = self.env.reset(obs_as_dict=False)
            state = np.asarray(state)
            self.noise.reset()

            for i_step in itertools.count():
                action = self.sess.run(self.model.Actor.output, feed_dict={
                    self.model.states_ph : np.expand_dims(state,0),
                    self.model.is_training_ph : False
                })[0]
                action += self.noise() * self.noise_decay
                # execute action action_with_noise and observe reward r_t and s_t+1
                state, reward, done, _ = self.env.step(action, obs_as_dict=False)

                reward = 0
                if terminal:
                    reward = self.tools.get_reward(self.direction, self.env.get_state_desc())
                name = "./log/training.txt"
                with open(name, 'a') as f:
                    f.write("Episode {}/{} == Step : {} =>>> Reward {} \n".format(i_episode + 1, self.num_episodes, i_step, reward))
                f.close()
                state = np.asarray(state)
                self.model.memory_buffer.add(action, reward, state, done)

                one_episode_score += reward

                self.experience_replay()

                if done or i_step == 50000:
                    end = time.time()
                    print("Episode {} =>>>>> Score {}".format(i_episode + 1, one_episode_score))
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
                    # timer
                    summary = self.sess.run(self.tf_step_summary, feed_dict={
                        self.tf_step: i_step
                    })
                    self.writer.add_summary(summary, i_episode)
                    break
            name = "./log/training.txt"
            with open(name, 'a') as f:
                f.write("Total score : {} \n".format(one_episode_score))
            f.close()

        # save information to log
        name = "./log/data.txt"
        with open(name, 'a') as f:
            f.write(" ".join((self.num_episodes, model_name_prefix, self.noise, self.epsilon)))
        f.close()
        print("Log saved successfully! \n")
        # save model
        self.saveWeights("./models/{}_{}_".format(args.direction, args.episodes))
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
        targets = batch_reward + (future_Q * self.model.discount)
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


if __name__ == '__main__':
    args = arg_parser()
    env = L2M2019Env(visualize=False)
    # Create session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model = Actor_Critic(env, args)
    print("======= Start Training =======\n")
    trainer = Trainer(model, env, sess, args)
    if trainer.tryLoadWeights() == 1:
        print("Play to initiate buffer !")
        trainer.play_to_init_buffer()

    trainer.DDPG(model_name_prefix="./models/" + args.direction)
    print("======= Training Completed =======\n")
