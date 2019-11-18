
#*********************************************************
#This is the class Actor_Critic for algorithmddpg actor_critic. 
#In this algorithm we use noise decay for exploration of agent.
#Noise is created by Gaussian distribution.
#
#Proprieties : 
#    - env : environment used for training
#    - sess: tensorflow session
#    - memory_buffer : memory buffer for experience replay
#    - Actor : actor network 
#    - Critic : critic network
#
#Methods :
#1) update_target : update the weights of target networks
#
#2) train : train actor network and critic network
#**********************************************************

import sys
sys.path.append("../")
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
from replay_buffer import Replay_Buffer
from actor import Actor
from critic import Critic
import utils
import argparse
import itertools

TAU = 0.125
LR = 0.001
BUFFER_SIZE = 10000
BATCH_SIZE = 32
DISCOUNT = 0.99
NOISE = 0.5
NOISE_DECAY = 0.99
EPSILON = 1
EPSILON_DECAY = 0.99

class Actor_Critic():
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess
        self.memory_buffer = Replay_Buffer(BUFFER_SIZE, BATCH_SIZE)
        self.learning_rate = LR
        self.tau = TAU
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.discount = 0.99
        self.Actor = Actor(self.env, self.sess, self.learning_rate, self.tau, self.discount)
        self.Critic = Critic(self.env, self.sess, self.learning_rate, self.tau, self.discount)

    def update_target(self):
        self.Actor.actor_target_update()
        self.Critic.critic_target_update()
    
    #def train(self):
        
    def save(self, prefixe):
        self.Actor.save(prefixe)
        self.Critic.save(prefixe)
        self.memory_buffer.save()

class Trainer():
    def __init__(self, model, num_episodes, direction):
        self.model = model
        self.direction = direction
        self.num_episodes = num_episodes
        self.episode_start = 0
        self.noise = NOISE
        self.noise_decay = nth_root(self.num_episodes, 0.01 / self.noise)
        self.epsilon = EPSILON
        self.epsilon_decay = nth_root(self.num_episodes, 0.01 / self.epsilon)
        self.count_exp_replay = 0
        # reward summary for tensorboard
        self.tf_reward = tf.placeholder(tf.float32, shape=None, name='reward_summary')
        self.tf_reward_summary = tf.summary.scalar("Reward by episode", self.tf_reward)
        # Loss of critic summary for tensorboard
        self.tf_critic_loss = tf.placeholder(tf.float32, shape=None, name='critic_loss_summary')
        self.tf_critic_summary = tf.summary.scalar("Critic loss", self.tf_critic_loss)
        # Loss of actor summary for tensorboard
        self.tf_actor_loss = tf.placeholder(tf.float32, shape=None, name='actor_loss_summary')
        self.tf_actor_summary = tf.summary.scalar("Actor loss", self.tf_actor_loss)
        # merge loss
        self.loss = tf.summary.merge([self.tf_critic_summary, self.tf_actor_summary])
        # time
        self.tf_time = tf.placeholder(tf.float32, shape=None, name='Time_per_episode')
        self.tf_time_summary = tf.summary.scalar("Time per episode", self.tf_time)
        # writer
        self.writer = tf.summary.FileWriter('./graphs', self.model.sess.graph)
    
    def tryLoadWeights(self):
        print("Load weights \n")
        try:
            with open("./log/data.txt", 'r') as f:
                data = f.read()
            num_episodes, model_name_prefix, noise , epsilon = data.split(" ")
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
        except:
            #if self.episode_start == 0:
            #    return(False)
            print("New training \n")
            return(1)
        return(True)

    def play_to_init_buffer(self):
        for i_episode in range(self.model.batch_size):
            #reset env
            state = env.reset(obs_as_dict = False)
            
            #noise decay
            self.noise = self.noise * self.noise_decay

            #epsilon decay
            self.epsilon = self.epsilon * self.epsilon_decay

            noise = np.zeros([1, self.model.env.action_space.shape[0]])

            action_with_noise = np.zeros([1, self.model.env.action_space.shape[0]])
            
            for i_step in itertools.count():
                state = [state]
                action_original = self.model.Actor.actor_model.predict(np.asarray(state))

                #action for training with noise
                action_with_noise[0] = np.zeros([1, self.model.env.action_space.shape[0]])

                #We use epsilon decay to decide wether we add noise to action or not
                s = np.random.binomial(1, self.epsilon)
                if s == 1:
                    #create noise by Gaussian distribution
                    noise[0] = np.random.randn(self.model.env.action_space.shape[0]) * self.noise
                action_with_noise = action_original + noise

                #execute action action_with_noise and observe reward r_t and s_t+1
                next_state, reward, done, info = self.model.env.step(action_with_noise[0], obs_as_dict=False)
                reward = -reward
                if done:
                    if utils.direction(next_state) == self.direction:
                        reward += 10
                    else:
                        reward -= 10
                
                self.model.memory_buffer.memorize([state, action_with_noise, reward, next_state, done])

                if done:
                    break
                else:
                    state = next_state

    def DDPG(self, model_name_prefix):
        scores = []
        for i_episode in range(self.episode_start, self.num_episodes):
            one_episode_score = 0
            #write log of training
            name = "./log/training.txt"
            with open(name, 'a') as f:
                f.write("Episode {}/{} \n".format(i_episode + 1, self.num_episodes))
            f.close()

            if (i_episode + 1) % 100 == 0:
                avg = np.mean(np.asarray(scores))
                if (i_episode + 1) % 1000 == 0:
                    prefixe = "./checkpoints/"
                    self.model.Actor.save(prefixe = prefixe + "checkpoint_avgScore_{}".format(avg))
                    self.model.Critic.save(prefixe = prefixe + "checkpoint_avgScore_{}".format(avg))
                print("Episode {}/{} : Average score in 100 latest episodes : {}".format(i_episode+1, self.num_episodes, avg))
                scores.clear()
            
            #reset env
            state = env.reset(obs_as_dict = False)
            
            #noise decay
            self.noise *= self.noise_decay

            #epsilon decay
            self.epsilon *= sself.epsilon_decay

            noise = np.zeros([1, self.model.env.action_space.shape[0]])

            action_with_noise = np.zeros([1, self.model.env.action_space.shape[0]])
            
            for i_step in itertools.count():
                state = [state]
                action_original = self.model.Actor.actor_model.predict(np.asarray(state))

                #action for training with noise
                action_with_noise[0] = np.zeros([1, self.model.env.action_space.shape[0]])

                #We use epsilon decay to decide wether we add noise to action or not
                s = np.random.binomial(1, self.epsilon)
                if s == 1:
                    #create noise by Gaussian distribution
                    noise[0] = np.random.randn(self.model.env.action_space.shape[0]) * self.noise
                action_with_noise = action_original + noise

                #execute action action_with_noise and observe reward r_t and s_t+1
                next_state, reward, done, info = self.model.env.step(action_with_noise[0], obs_as_dict=False)
                reward = -reward
                if done:
                    if utils.direction(next_state) == self.direction:
                        reward += 10
                    else:
                        reward -= 10
                one_episode_score += reward
                self.model.memory_buffer.memorize([state, action_with_noise, reward, next_state, done])
                self.experience_replay()

                if done:
                    print("Episode {} ->>> Scores : {}".format(i_episode, one_episode_score))
                    scores.append(one_episode_score)
                    # write reward for tensorboard
                    summary = self.model.sess.run(self.tf_reward_summary, feed_dict={
                        self.tf_reward: one_episode_score
                    })
                    # add summary to writer
                    self.writer.add_summary(summary, i_episode)
                    break
                else:
                    state = next_state
            name = "./log/training.txt"
            with open(name, 'a') as f:
                f.write("Total score : {} \n".format(one_episode_score))
            f.close()

        #save information to log
        name = "./log/data.txt"
        with open(name, 'a') as f:
            f.write(" ".join((self.num_episodes, model_name_prefix, self.noise, self.epsilon)))
        f.close()
        print("Log saved successfully! \n")
        #save memory buffer
        self.model.memory_buffer.save()
        print("Memory buffer saved successfully \n")
        #save model
        model.save("./models/{}_{}".format(args.direction, args.episodes))
        print("Models saved successfully ! \n")
        
    def experience_replay(self):
        #hist_actor, hist_critic = self.model.train()
        samples = self.model.memory_buffer.sample_batch() 
        history_critic = self.model.Critic.critic_train(self.model.Actor.actor_target, samples)
        action_for_grad = self.model.Actor.actor_model.predict(samples[0])
        grad = self.model.Critic.gradients(samples[0], action_for_grad)
        history_actor = self.model.Actor.actor_train(grad, samples)
        
        self.model.update_target()
        #write log
        name = "./log/training.txt"
        with open(name, 'a') as f:
            f.write("History of actor network : {} \n".format(history_actor))
            f.write("History of critic network : {} \n".format(history_critic))
        f.close()

        # write loss for tensorboard
        summary = self.model.sess.run(self.loss, feed_dict={
            self.tf_critic_loss: history_critic,
            self.tf_actor_loss: history_actor
        })

        # add summary to writer
        self.writer.add_summary(summary, self.count_exp_replay)
        self.count_exp_replay += 1

        return([history_actor, history_critic])
        
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", default = "forward", help = "direction of falling")
    parser.add_argument("--out", default = "front", help = "prefix of output file")
    parser.add_argument("--episodes", default = 10000, help = "number of episodes for training")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parser()
    env = L2M2019Env(visualize=False)
    sess = tf.Session()
    model = Actor_Critic(env, sess)
    print("======= Start Training =======\n")
    trainer = Trainer(model, args.episodes, args.direction)
    if trainer.tryLoadWeights() == 1:
        trainer.play_to_init_buffer()
    trainer.DDPG(model_name_prefix = "./models/"+args.direction)
    print("======= Training Completed =======\n")
