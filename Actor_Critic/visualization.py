import sys
sys.path.append("../")

from osim.env import L2M2019Env
import numpy as np
import tensorflow as tf
from src.actor import ActorNetwork
import itertools
from Actor_Critic.train import Actor_Critic
import argparse
from src.ou_noise import OUNoise
from src.utils import Tools

import json
import plotly.graph_objects as go
import os
import time

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./checkpoints/left_6999.ckpt", help="path to checkpoint .cKpt")
    parser.add_argument("--direction", default="left", help="direction of falling")
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

class Visualisation:
    def __init__(self, env, arg, model_path, save_data_path):
        self.env = env
        self.arg = arg
        self.model_path = model_path
        self.save_data_path = save_data_path
        self.noise = OUNoise(mu=np.zeros(self.env.action_space.shape))
        self.tools = Tools()
        self.direction = arg.direction


    def muscle_visualisation(self, path):
        cwd = os.getcwd()  # Get the current working directory (cwd)
        files = os.listdir(cwd)  # Get all the files in that directory
        print("Files in %r: %s" % (cwd, files))

        with open(path) as json_file:
            fig = go.Figure()
            dataset = json.load(json_file)
            a = dataset["episode0"]
            a = np.array(a)
            a = np.transpose(a)
            fig.add_trace(go.Heatmap(
                z=a,
                colorscale=[[1, "rgb(255,69,0)"],
                            [0, "rgb(255,255,255)"]]
            ))

            fig.update_layout(
                title="Muscle Visualisation",
                xaxis_title="Time step",
                yaxis_title="Valeur"
            )

            fig.show()

    # get the angles of three joints on each leg:
    # return array of rhip, rknee, rankle, lhip, lknee, lankle
    def get_angles(self, state_desc):
        body_pos = state_desc["joint_pos"]
        angles = []
        angles.append(body_pos["hip_r"])
        angles.append(body_pos["knee_r"])
        angles.append(body_pos["ankle_r"])
        angles.append(body_pos["hip_l"])
        angles.append(body_pos["knee_l"])
        angles.append(body_pos["ankle_l"])
        return angles

    def articulation_visualisation(self, path):
        cwd = os.getcwd()  # Get the current working directory (cwd)
        files = os.listdir(cwd)  # Get all the files in that directory
        print("Files in %r: %s" % (cwd, files))
        with open(path) as json_file:
            fig = go.Figure()
            step = itertools.count()
            dataset = json.load(json_file)["episode0"]
            step = np.arange(0, len(dataset), 1)
            hip_r_x = []
            hip_r_y = []
            hip_r_z = []
            knee_r = []
            ankle_r = []
            hip_l_x = []
            hip_l_y = []
            hip_l_z = []
            knee_l = []
            ankle_l = []
            for angles in dataset:
                hip_r_x.append(angles[0][0])
                hip_r_y.append(angles[0][1])
                hip_r_z.append(angles[0][2])
                knee_r.append(angles[1][0])
                ankle_r.append(angles[2][0])
                hip_l_x.append(angles[3][0])
                hip_l_y.append(angles[3][1])
                hip_l_z.append(angles[3][2])
                knee_l.append(angles[4][0])
                ankle_l.append(angles[5][0])

            fig.add_trace(go.Scatter(x=step, y=hip_r_x,
                                     mode='lines',
                                     name='hip_r_x'))
            fig.add_trace(go.Scatter(x=step, y=hip_r_y,
                                     mode='lines',
                                     name='hip_r_y'))
            fig.add_trace(go.Scatter(x=step, y=hip_r_z,
                                     mode='lines',
                                     name='hip_r_z'))
            fig.add_trace(go.Scatter(x=step, y=knee_r,
                                     mode='lines',
                                     name='knee_r'))
            fig.add_trace(go.Scatter(x=step, y=ankle_r,
                                     mode='lines',
                                     name='ankle_r'))

            fig.add_trace(go.Scatter(x=step, y=hip_l_x,
                                     mode='lines',
                                     name='hip_l_x'))
            fig.add_trace(go.Scatter(x=step, y=hip_l_y,
                                     mode='lines',
                                     name='hip_l_y'))
            fig.add_trace(go.Scatter(x=step, y=hip_l_z,
                                     mode='lines',
                                     name='hip_l_z'))
            fig.add_trace(go.Scatter(x=step, y=knee_l,
                                     mode='lines',
                                     name='knee_l'))
            fig.add_trace(go.Scatter(x=step, y=ankle_l,
                                     mode='lines',
                                     name='ankle_l'))

            fig.update_layout(
                title="Articulation Visualisation",
                xaxis_title="Time step",
                yaxis_title="Valeur"
            )

            fig.show()

    def load_model(self, env, args):
        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.model = Actor_Critic(env, args)
            # print(self.graph.get_operations())
            #saver = tf.train.import_meta_graph("../Actor_Critic/checkpoints/forward_9999_model.ckpt.meta")
            saver = tf.train.Saver()
            saver.restore(sess, args.checkpoint)
            print("Load successful ! ")

            for i_episode in range(1):
                state = self.env.reset(obs_as_dict=False)
                state = np.asarray(state)
                self.noise.reset()
                one_episode_score = 0
                muscles = []
                angles = []
                angle_state = np.arccos(self.tools.get_reward(self.direction, self.env.get_state_desc()))
                for i_step in itertools.count():
                    time.sleep(0.2)

                    action = sess.run(self.model.Actor.output, feed_dict={
                        self.model.states_ph: np.expand_dims(np.array([angle_state]), 0)
                    })[0]
                    # execute action action_with_noise and observe reward r_t and s_t+1
                    next_state, reward, done, _ = self.env.step(action, obs_as_dict=False)
                    state_desc = env.get_state_desc()
                    muscles_desc = state_desc["muscles"]
                    muscles_activation = []
                    for muscle in muscles_desc:
                        muscles_activation.append(muscles_desc.get(muscle)["activation"])
                    muscles.append(muscles_activation)
                    angles.append(self.get_angles(state_desc))
                    reward = self.tools.get_reward(self.direction, self.env.get_state_desc())
                    next_state = np.asarray(next_state)
                    state = np.copy(next_state)
                    angle_next_state = np.arccos(self.tools.get_reward(self.direction, self.env.get_state_desc()))
                    angle_state = angle_next_state

                    print("Time step {} test {} =>>>>>>> reward {} ".format(i_step, i_episode, reward))
                    one_episode_score += reward

                    if done or i_step == 50000:
                        print("Episode {} =>>>>> Score {}".format(i_episode + 1, one_episode_score))
                        break
                muscles = {"episode" + str(i_episode): muscles}
                angles = {"episode" + str(i_episode): angles}
                with open(self.save_data_path + "left_3999_model_decalage.json", 'w') as f:
                    json.dump(muscles, f)
                with open(self.save_data_path + "articulation.json", 'w') as f:
                    json.dump(angles, f)
            sess.close()

    def run_model(self):
        return 0

if __name__ == '__main__':

    args = arg_parser()
    env = L2M2019Env(visualize=True)
    visualiser = Visualisation(env, args, "./checkpoints", "./checkpoints/")
    visualiser.load_model(env,args)
    #visualiser.muscle_visualisation('../Actor_Critic/log/action.json')
    visualiser.muscle_visualisation('./checkpoints/left_3999_model_decalage.json')
    visualiser.articulation_visualisation('./checkpoints/articulation.json')