import numpy as np
from PartLy.core import World, Agent, KeyItem
from PartLy.scenario import BaseScenario


# Work in progress for supporting a scheduler and a partitioner with different observation and action spaces

class Scenario(BaseScenario):
    def make_world(self):
        # TODO
        world = World()
        world.maxKeys = 1000
        num_good_agents = 2
        num_scheduler = 4
        num_agents = num_scheduler + num_good_agents
        num_keys = 1000
        num_blocks = 50
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        # add keys
        world.keys = [KeyItem() for i in range(num_keys)]
        # make initial conditions
        self.reset_world(world)
        return world




    def reset_world(self, world):
        # TODO
        # reset properties for agents
        return


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each key
        #boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = self.scheduler_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward


    def partitioner_reward(self, agent, world):
        # TODO

        return 1

    def scheduler_reward(self, agent, world):
        # TODO

        return 1


    def partitioner_observation(self, agent, world):
        # TODO
        return 0

    def scheduler_observation(self, agent, world):
        # TODO
        return 0




