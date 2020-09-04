import numpy as np
from PartLy.core import World, Agent, KeyItem
from PartLy.scenario import BaseScenario

# PartLy: Assume a single agent (i.e., one data partitioner)
class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.split = False
            agent.silent = True
        # add keys
        world.keys = [KeyItem() for i in range(1)]
        for i, key in enumerate(world.keys):
            key.name = 'key %d' % i
            key.split = False
            key.assign = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # set random initial states
        for agent in world.agents:
            agent.state.blocks = np.random.uniform(-1, +1, world.maxBlocks)
            agent.state.keys = np.zeros(world.maxKeys)
            agent.state.c = np.zeros(world.maxBlocks)
        for i, key in enumerate(world.keys):
            key.state.blocks = np.random.uniform(-1, +1, world.dim_p)
            key.state.keys = np.zeros(world.dim_p)

    def reward(self, agent, world):
        balance = np.subtract(np.square(agent.state.blocks - world.keys[0].state.blocks))
        return -balance

    def observation(self, agent, world):
        assinged = []
        for entity in world.keys:
            assinged.append(entity.state.blocks - agent.state.blocks)
        return np.concatenate([agent.state.keys] + assinged)
