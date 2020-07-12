from gym.envs.registration import register

register(
        id='metak-v0',
        entry_point='gym_metak.envs:MetakEnv',
)
