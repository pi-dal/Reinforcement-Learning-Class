from gymnasium.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="reinforcement_learning_class.env.gridworld_env:GridWorldEnv",
)
