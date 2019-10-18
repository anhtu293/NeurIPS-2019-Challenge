from osim.env import L2M2019Env
from utils import Tools

env = L2M2019Env(visualize=True)
observation = env.reset(project=True, seed=None, init_pose=None, obs_as_dict=True)


# print(env.get_observation_dict())

def init():
    tools = Tools()
    for i in range(200):
        observation, reward, done, info = env.step(tools.active_one_muscle("rect_fem", "r", 1))
        if i == 20:
            input("Press Enter to continue...")
            print(reward)


if __name__ == '__main__':
   init()
