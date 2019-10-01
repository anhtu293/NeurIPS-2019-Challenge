from osim.env import L2M2019Env

env = L2M2019Env(visualize=True)
observation = env.reset()
print(env.get_observation_dict())
"""
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    print(observation)

"""