import numpy as np
from pyglet.window import key

# Example individual agent policy
class Policy(object):
    def __init__(self):
        pass
    def action(self, obs):
        raise NotImplementedError()
#Sample Ad hoc policy, random assignment or split
class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.target = [False for i in range(env.world.maxBlocks)]
        self.input = [False for i in range(env.world.maxKeys)]

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.target[0]: u = 1
            if self.target[1]: u = 2
            if self.target[2]: u = 4
            if self.target[3]: u = 3
        else:
            u = np.zeros(50)
            if self.target[0]: u[1] += 1.0
            if self.target[1]: u[2] += 1.0
            if self.target[3]: u[3] += 1.0
            if self.target[2]: u[4] += 1.0
            if True not in self.target:
                u[0] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.maxKeys)])