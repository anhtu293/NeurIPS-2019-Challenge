import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity

# Tools for training
# 1) direction():

class Tools:
    dict_muscle = {'abd': 'HAB',
                   'add': 'HAD',
                   'iliopsoas': 'HFL',
                   'glut_max': 'GLU',
                   'hamstrings': 'HAM',
                   'rect_fem': 'RF',
                   'vasti': 'VAS',
                   'bifemsh': 'BFSH',
                   'gastroc': 'GAS',
                   'soleus': 'SOL',
                   'tib_ant': 'TA'}

    def active_one_muscle(self, name, leg, excitation):
        pos = 0
        activations = np.zeros(22)
        for MUS, mus in self.dict_muscle.items():
            if name == MUS or name == mus:
                if leg == "r":
                    activations[pos] = excitation
                elif leg == "l":
                    activations[pos + 11] = excitation
                else:
                    print("Wrong leg")
                    return
            pos += 1
        return activations

    def get_direction(self, toes_l, toes_r, mass_center_pos):
        l = np.asarray(toes_l)
        r = np.asarray(toes_r)
        mp = np.asarray((l + r) / 2)
        mass_center_pos = np.asarray(mass_center_pos)
        body_direction = mass_center_pos - mp

        return body_direction

    def get_reward(self, direction, state_desc):
        pl = state_desc["body_pos"]["talus_l"]
        pr = state_desc["body_pos"]["talus_r"]
        ms = state_desc["misc"]["mass_center_pos"]

        center_2_feet = np.asarray((np.asarray(pl) + np.asarray(pr)) / 2)
        if direction == "forward":
            center_2_feet[0] = 1
        if direction == "left":
            center_2_feet[2] = 1
        if direction == "right":
            center_2_feet[1] = 1
        direction_real = self.get_direction(pl, pr, ms)

        direction_real = np.asarray([direction_real])
        direction_to_fall = np.asarray([center_2_feet])
        penalty = cosine_similarity(direction_real, direction_to_fall)[0][0]
        #penalty = np.pi - np.arccos(cosine)
        return penalty

if __name__ == '__main__':
    tools = Tools()
    #print(tools.active_one_muscle("rect_fem", "r", 1))
    a = math.sqrt(3)
    print(tools.get_direction([-1, 0], [1, 0], [a, -1, 0]))
