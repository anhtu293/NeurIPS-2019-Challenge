import numpy as np


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



if __name__ == '__main__':
    tools = Tools()
    print(tools.active_one_muscle("rect_fem", "r", 1))
