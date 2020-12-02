# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:57:29 2020

@author: Venkata Harshit Koneru
"""

import matplotlib.pyplot as plt
from actor_critic_cart_pole import Actor
from actor_critic_cart_pole import Critic
import gym
import numpy as np

env_name="CartPole-v0"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = 2

k_ep = np.linspace(0, 9999, num=10000)
Actor_LR = 0.0001
Critic_LR= 0.0001

GAMMA = 0.999
max_episodes=10000
rewardCal = np.zeros([10000, 1])
trainingTime = np.zeros([10000, 1])
Actor_A2C = Actor(state_dim, action_dim, Actor_LR)
Critic_A2C = Critic(Critic_LR)


meanrewards=np.zeros(max_episodes)
total_episode_reward=[]


for episode in range(max_episodes):
    
     done = False
     total_reward = 0
     trainTime = 0
     observation=env.reset()
     done = False
     while not done:
         
            
            action =Actor_A2C.choose_action(observation)
            act=action.item()
            observation_, reward,done,_ = env.step(act)
            td_error = Critic_A2C.learn(observation, reward,observation_, GAMMA)
            Actor_A2C.learn(observation, action, td_error)
            observation = observation_
            total_reward += reward
            
            
     print("Epiosde {} has a total reward of {}".format(episode,total_reward)) 
     total_episode_reward.append(total_reward)
     rewardCal[episode] = total_reward
     


plt.figure()
plt.plot(k_ep, rewardCal)
plt.xlabel('episode')
plt.ylabel('Sumreward at each episode')




