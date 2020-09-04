# PartLy

A learned data partitioner for streaming workloads that relies on Deep Reinforcement Learning. It assumes a micro-batch oriented execution of the incoming data stream tuples.
It is built on an environment that support discrete action space for either assigning or splitting data keys over a set of partitions. 
The reward of the training episodes is calculated based on the size-equality of the generated data partitions.  


## Getting started:

- Dependencies: Python (3.5.4), OpenAI gym (0.10.5), TensorForce (0.5.4), numpy (1.14.5)

- To use the environments, look at the code for importing them in `make_env.py`.

## Code structure

To run this version of Partly. Run main.py for example of training the agent using 20,000 batches and then testing using 1000 batches. Note that each batch correspond to an episode in training or evaluation. We are working on add more features and improving the action space.

## Code structure

- `make_env.py`: contains code for importing the environment as an OpenAI Gym-like object.

- `./PartLy/environment.py`: contains code for environment simulation 

- `./PartLy/core.py`: contains classes for various objects (keys, Agents, etc.) that are used throughout the code.

- `./PartLy/scenario.py`: contains base scenario object that is extended for all scenarios.

- `./PartLy/scenarios/`: folder where various environments are stored. scenario code consists of several functions:
    1) `make_world()`: creates all of the entities that in the environment (keys, agents, etc.), called once at the beginning of each training session
    2) `reset_world()`: resets the environment to default properties. called before every episode (including after make_world() before the first episode)
    3) `reward()`: defines the reward function for a given agent
    4) `observation()`: defines the observation space of a given agent

### Creating new environments

You can create new data partitioning agents by implementing the first 4 functions above (`make_world()`, `reset_world()`, `reward()`, and `observation()`).

## List of Agents

We are current working to support the below agents:

| Env name in code |  Cost Model | Notes |
| --- | --- | --- | 
| `Partitioner_PK2.py` | PK2 | Single agent partitions a list of  keys over a specificed number of data blocks. It currently support either two actions types (assign and split). |
| `Scheduler_Partitoner.py` | TBA | 1 partitioner , 1 scheduler. Both agents are work cooperatively to decide partitioning and parallelism -- Work in Progress. |

## Publications

* Ahmed S. Abdelhamid and Walid G. Aref, “PartLy: Learning Data Partitioning for Distributed Data Stream Processing”, In Proceedings of 3rd International Workshop on Exploiting Artificial Intelligence Techniques for Data Management, June 19, 2020 (Short Paper)

## Contact

If you have any question please feel free to send an email.

* Ahmed S. Abdelhamid <samy@purdue.edu> 