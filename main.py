# Copyright 2018 Tensorforce Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import logging

import tensorflow as tf

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def main():
    # Create an OpenAI-Gym environment
    environment = Environment.create(environment='gym', level='CartPole-v1', max_episode_timesteps=500)

    # Create a PPO agent with training parameters
    agent = Agent.create(
        agent='ppo', environment=environment,
        # Automatically configured network
        network='auto',
        # Optimization
        batch_size=1000, update_frequency=2, learning_rate=1e-3, subsampling_fraction=0.2,
        optimization_steps=5,
        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=0.99, estimate_terminal=False,
        # Critic
        critic_network='auto',
        critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
        # Preprocessing
        preprocessing=None,
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # TensorFlow etc
        name='PartLy', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
        summarizer=None, recorder=None
    )

    # Initialize the runner
    runner = Runner(agent=agent, environment=environment)

    # Train for 20000 episodes
    for episode in range(20000):

        # Record episode experience
        episode_states = list()
        episode_internals = list()
        episode_actions = list()
        episode_terminal = list()
        episode_reward = list()

        # Episode using independent-act and agent.intial_internals()
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        sum_reward = 0.0
        while not terminal:
            episode_states.append(states)
            episode_internals.append(internals)
            actions, internals = agent.act(states=states, internals=internals, independent=True)
            episode_actions.append(actions)
            states, terminal, reward = environment.execute(actions=actions)
            episode_terminal.append(terminal)
            episode_reward.append(reward)
            sum_reward += reward
        print('Episode {}: {}'.format(episode, sum_reward))

        # Record experience to agent
        agent.experience(
            states=episode_states, internals=episode_internals, actions=episode_actions,
            terminal=episode_terminal, reward=episode_reward
        )

        # Perform update
        agent.update()

    # Testing for 1000 episodes/batches
    sum_rewards = 0.0
    for _ in range(1000):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent.act(
                states=states, internals=internals, independent=True, deterministic=True
            )
            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward
    print('Mean evaluation return:', sum_rewards / 100.0)

    # # Start the runner
    # runner.run(num_episodes=200)
    # runner.close()



if __name__ == '__main__':
    main()
