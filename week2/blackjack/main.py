import gymnasium as gym

env = gym.make('Blackjack-v1', natural=False, sab=False)

obs, info = env.reset()

print(obs, info, env)

obs, info = env.reset()

print(obs, info, env)