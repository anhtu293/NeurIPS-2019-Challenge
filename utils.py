import numpy as np
import math

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
# Counterclockwise
    def get_direction(self, toes_l, toes_r, mass_center_pos):
        l = np.asarray(toes_l)
        r = np.asarray(toes_r)
        mp = np.asarray((l + r)/2)
        mass_center_pos = np.asarray(mass_center_pos)
        l_to_r = r - l
        rot = np.pi/2
        m_t = np.zeros((2, 2))
        m_t[0][0] = np.cos(rot)
        m_t[0][1] = -np.sin(rot)
        m_t[1][0] = np.sin(rot)
        m_t[1][1] = -np.cos(rot)
        straight_direction = np.dot(m_t, l_to_r)
        body_direction = mass_center_pos - mp
        theta = math.atan2(body_direction[1], body_direction[0]) - \
                math.atan2(straight_direction[1], straight_direction[0])
        theta = theta * 180 / np.pi
        return theta

if __name__ == '__main__':
    tools = Tools()
    #print(tools.active_one_muscle("rect_fem", "r", 1))
    a = math.sqrt(3)
    print(tools.get_direction([-1, 0], [1, 0 ], [a, -1]))
