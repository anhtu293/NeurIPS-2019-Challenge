from osim.env import L2M2019Env
from utils import Tools

env = L2M2019Env(visualize=True)
observation = env.reset(project=True, seed=None, init_pose=None, obs_as_dict=True)


# print(env.get_observation_dict())

def init():
    tools = Tools()

    for i in range(200):
        observation, reward, done, info = env.step(tools.active_one_muscle("iliopsoas", "r", 1))

        if i == 20:
            state_desc = env.get_state_desc()
            print(type(state_desc["body_pos"]["toes_r"][0:2]))
            print(state_desc["body_pos"]["talus_l"][0:2])
            print(state_desc["misc"]["mass_center_pos"])
            print(state_desc["body_pos_rot"])
            input("Press Enter to continue...")
            print(reward)


if __name__ == '__main__':
    print(env.action_space.high)
    print(env.action_space.low)

