from gym.envs.registration import register

register(
    id='PartLySimple-v0',
    entry_point='PartLy.envs:PartitonerEnv',
    max_episode_steps=1000,
)

register(
    id='PartLyMultiSchedulerPartitoner-v0',
    entry_point='PartLy.envs:SchedulerPartitonerEnv',
    max_episode_steps=1000,
)
