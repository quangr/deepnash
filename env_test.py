import gym
import gym_kuhn_poker
env = gym.make('KuhnPoker-v0', **dict()) 
print(env.reset())