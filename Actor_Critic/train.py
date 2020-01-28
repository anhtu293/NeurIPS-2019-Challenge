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
from src.actor import ActorNetwork
from src.critic import CriticNetwork
from src.ou_noise import OUNoise
from src.utils import Tools
import argparse
import itertools
import time
from src.replay_buffer import ReplayBuffer


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
        self.states_ph = tf.placeholder(tf.float32, shape=(None,1))
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

class Trainer():
    def __init__(self, env,  args):
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
        state = self.env.reset(obs_as_dict=False)
        state = np.asarray(state)
        for random_step in range(1, args.init_buffer_size + 1):
            #self.env.render()
            print("\r Examples : {}/{}".format(random_step, args.init_buffer_size), end="")
            sys.stdout.flush()
            action = self.env.action_space.sample()
            next_state, reward, terminal, _ = self.env.step(action, obs_as_dict=False)

            reward = self.tools.get_reward(self.direction, self.env.get_state_desc())

            next_state = np.asarray(next_state)
            self.model.memory_buffer.add(state, action, reward, next_state, terminal)
            state = np.copy(next_state)
            if terminal:
                self.env.reset()

    def DDPG(self):

        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.model = Actor_Critic(env, args)
            self.target_Q_ph = tf.placeholder(tf.float32, shape=(None, 1))
            self.actions_grads_ph = tf.placeholder(tf.float32, shape=((None,) + self.env.action_space.shape))
            # train operation
            self.actor_train_ops = self.model.Actor.train_step(self.actions_grads_ph)
            self.critic_train_ops = self.model.Critic.train_step(self.target_Q_ph)
            # update operation
            self.update_critic_target = self.model.update_target_network(self.model.Critic.network_params,
                                                                         self.model.Critic_target.network_params,
                                                                         self.tau)
            self.update_actor_target = self.model.update_target_network(self.model.Actor.network_params,
                                                                        self.model.Actor_target.network_params,
                                                                        self.tau)
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
            self.writer = tf.summary.FileWriter('./graphs', sess.graph)
            sess.run(tf.initialize_all_variables())
            # init target networks by copying weights from actor and critic network
            sess.run(
                self.model.update_target_network(self.model.Critic.network_params, self.model.Critic_target.network_params))
            sess.run(
                self.model.update_target_network(self.model.Actor.network_params, self.model.Actor_target.network_params))
            #save by saver
            saver = tf.train.Saver()

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
                    self.noise_decay *= 0.95
                    if (i_episode + 1) % 500 == 0:
                        #self.saveWeights("./checkpoints/{}_{}_".format(args.direction, i_episode), sess)
                        save = saver.save(sess, "./checkpoints/left_{}.ckpt".format(i_episode))
                        print(save)
                    print(
                        "Episode {}/{} : Average score in 100 latest episodes : {}".format(i_episode + 1, self.num_episodes,
                                                                                           avg))
                    scores.clear()
                # reset env
                state = self.env.reset(obs_as_dict=False)
                state = np.asarray(state)
                self.noise.reset()
                angle_state = np.arccos(self.tools.get_reward(self.direction, self.env.get_state_desc()))

                for i_step in itertools.count():
                    action = sess.run(self.model.Actor.output, feed_dict={
                        self.model.states_ph : np.expand_dims(np.array([angle_state]),0),
                        self.model.is_training_ph : False
                    })[0]

                    action += self.noise() * self.noise_decay

                    # execute action action_with_noise and observe reward r_t and s_t+1
                    next_state, reward, terminal, _ = self.env.step(action, obs_as_dict=False)
                    #reward = self.tools.get_reward(self.direction, self.env.get_state_desc())
                    angle_next_state = np.arccos(self.tools.get_reward(self.direction, self.env.get_state_desc()))
                    reward = np.cos(angle_next_state)

                    name = "./log/training.txt"
                    with open(name, 'a') as f:
                        f.write("Episode {}/{} == Step : {} =>>> Reward {} \n".format(i_episode + 1, self.num_episodes, i_step, reward))
                    f.close()

                    next_state = np.asarray(next_state)
                    self.model.memory_buffer.add(angle_state, action, reward, angle_next_state, terminal)
                    angle_state = angle_next_state

                    one_episode_score += reward
                    state = np.copy(next_state)
                    #self.experience_replay()

                    if self.model.memory_buffer.count() >= self.model.batch_size * 10:
                        batch, w_id, eid = self.model.memory_buffer.getBatch(self.model.batch_size)

                        batch_state = np.zeros((self.model.batch_size, 1))
                        batch_reward = np.zeros((self.model.batch_size,))
                        batch_action = np.zeros((self.model.batch_size, self.env.action_space.shape[0]))
                        batch_next_state = np.zeros((self.model.batch_size, 1))
                        batch_done = np.zeros((self.model.batch_size,))
                        e_id = eid

                        for k, (s0, a, r, s1, done) in enumerate(batch):
                            batch_state[k] = s0
                            batch_reward[k] = r
                            batch_action[k] = a
                            batch_next_state[k] = s1
                            batch_done[k] = done
                        batch_done = batch_done.astype(int)

                        future_action = sess.run(self.model.Actor_target.output, feed_dict={
                            self.model.states_ph: batch_next_state
                        })
                        future_Q = sess.run(self.model.Critic_target.output, feed_dict={
                            self.model.states_ph: batch_next_state,
                            self.model.actions_ph: future_action
                        })[:, 0]

                        future_Q[batch_done] = 0
                        targets = batch_reward + (future_Q * self.model.discount)
                        # train Critic
                        sess.run(self.critic_train_ops, feed_dict={
                            self.model.states_ph: batch_state,
                            self.model.actions_ph: batch_action,
                            self.target_Q_ph: np.expand_dims(targets, 1)
                        })

                        # train Actor
                        actor_actions = sess.run(self.model.Actor.output, feed_dict={
                            self.model.states_ph: batch_state
                        })
                        action_grads = sess.run(self.model.Critic.action_grads, feed_dict={
                            self.model.states_ph: batch_state,
                            self.model.actions_ph: actor_actions
                        })
                        sess.run(self.actor_train_ops, feed_dict={
                            self.model.states_ph: batch_state,
                            self.actions_grads_ph: action_grads[0]
                        })
                        # update target
                        sess.run(self.update_critic_target)
                        sess.run(self.update_actor_target)

                        # calcul TD error
                        old_Q_value = sess.run(self.model.Critic.output, feed_dict={
                            self.model.states_ph: batch_state,
                            self.model.actions_ph: batch_action
                        })[:, 0]
                        future_action = future_action = sess.run(self.model.Actor_target.output, feed_dict={
                            self.model.states_ph: batch_next_state
                        })
                        future_Q_value = sess.run(self.model.Critic_target.output, feed_dict={
                            self.model.states_ph: batch_next_state,
                            self.model.actions_ph: future_action
                        })[:, 0]
                        error = np.absolute(batch_reward + self.model.discount * (future_Q_value - old_Q_value))

                        # update priority
                        self.model.memory_buffer.update_priority(e_id, error)
                        self.train_iteration += 1
                        if self.train_iteration % 100 == 0:
                            self.model.memory_buffer.rebalance()

                    if terminal or i_step == 50000:
                        end = time.time()
                        print("Episode {} =>>>>> Score {}".format(i_episode + 1, one_episode_score))
                        scores.append(one_episode_score)
                        # write reward for tensorboard
                        summary = sess.run(self.tf_reward_summary, feed_dict={
                            self.tf_reward: one_episode_score
                        })
                        # add summary to writer
                        self.writer.add_summary(summary, i_episode)
                        # timer
                        summary = sess.run(self.tf_time_summary, feed_dict={
                            self.tf_time: end - start
                        })
                        self.writer.add_summary(summary, i_episode)
                        # timer
                        summary = sess.run(self.tf_step_summary, feed_dict={
                            self.tf_step: i_step
                        })
                        self.writer.add_summary(summary, i_episode)
                        break
                name = "./log/training.txt"
                with open(name, 'a') as f:
                    f.write("Total score : {} \n".format(one_episode_score))
                f.close()
            sess.close()

    """def experience_replay(self):
        #batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.model.memory_buffer.getMinibatch()

        if self.model.memory_buffer.count() < self.model.batch_size * 5:
            return
        batch, w_id, eid = self.model.memory_buffer.getBatch(self.model.batch_size)

        batch_state = np.zeros((self.model.batch_size, self.env.observation_space.shape[0]))
        batch_reward = np.zeros((self.model.batch_size,))
        batch_action = np.zeros((self.model.batch_size, self.env.action_space.shape[0]))
        batch_next_state = np.zeros((self.model.batch_size, self.env.observation_space.shape[0]))
        batch_done = np.zeros((self.model.batch_size,))
        e_id = eid

        for k, (s0, a, r, s1, done) in enumerate(batch):
            batch_state[k] = s0
            batch_reward[k] = r
            batch_action[k] = a
            batch_next_state[k] = s1
            batch_done[k] = done
        batch_done = batch_done.astype(int)

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

        #calcul TD error
        old_Q_value = self.sess.run(self.model.Critic.output, feed_dict={
            self.model.states_ph: batch_state,
            self.model.actions_ph: batch_action
        })[:,0]
        future_action = future_action = self.sess.run(self.model.Actor_target.output, feed_dict={
            self.model.states_ph : batch_next_state
        })
        future_Q_value = self.sess.run(self.model.Critic_target.output, feed_dict={
            self.model.states_ph : batch_next_state,
            self.model.actions_ph : future_action
        })[:,0]
        error = np.absolute(batch_reward + self.model.discount*(future_Q_value - old_Q_value))

        #update priority
        self.model.memory_buffer.update_priority(e_id, error)
        self.train_iteration += 1
        if self.train_iteration % 100 == 0:
            self.model.memory_buffer.rebalance()
"""
if __name__ == '__main__':
    args = arg_parser()
    env = L2M2019Env(visualize=False)
    # Create session
    print("======= Start Training =======\n")
    trainer = Trainer(env, args)
    trainer.DDPG()
    print("======= Training Completed =======\n")
