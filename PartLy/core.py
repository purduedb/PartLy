import numpy as np

class EntityState(object):
    def __init__(self):
        # data blocks
        self.blocks = None
        # keys assignents
        self.keys = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # split action
        self.u = None
        # assign action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.splitRatio = 0.50
        self.assign = True
        self.split = True

        self.state = EntityState()

        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of key entities
class KeyItem(Entity):
     def __init__(self):
        super(KeyItem, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.maxKeys = 1000
        self.maxBlocks = 50


    @property
    def entities(self):
        return self.agents #+ self.keys

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        newstep = [None] * len(self.entities)
        newstep = self.apply_action_force(newstep)

        self.integrate_state(newstep)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action
    # Exploration vs. Exploitation
    def apply_action_force(self, newstep):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.assign:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                newstep[i] = agent.action.u + noise
        return newstep

    # integrate state
    def integrate_state(self, newstep):
        for i,entity in enumerate(self.entities):
            if not entity.assign: continue
            entity.state.keys = entity.state.keys * (1 - self.damping)
            if (newstep[i] is not None):
                entity.state.keys += (newstep[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.keys[0]) + np.square(entity.state.keys[1]))
                if speed > entity.max_speed:
                    entity.state.keys = entity.state.keys / np.sqrt(np.square(entity.state.keys[0]) +
                                                                    np.square(entity.state.keys[1])) * entity.max_speed
            entity.state.blocks += entity.state.keys * self.dt

    def update_agent_state(self, agent):
            agent.state.c = np.zeros(self.maxKeys)
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      
