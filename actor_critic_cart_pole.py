# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 10:05:14 2020

@author:Venkata Harshit Koneru
"""

import gym
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.functional as Functions
import numpy as np
import matplotlib.pyplot as plt





class Actornetwork(nn.Module):
    'Actors NN such as the input,hidden_layersize,output has been defined in the constructer'
    def __init__(self, n_action):
       
        super(Actornetwork, self).__init__()
        self.action_space = 2
        self.layerInput = nn.Linear(4, 64)
        self.layerInput.weight.data.normal_(0, 0.1)
        self.HiddenLayer = nn.Linear(64, 64)
        self.HiddenLayer.weight.data.normal_(0, 0.1)
        self.layerOutput = nn.Linear(64, self.action_space)
        self.layerOutput.weight.data.normal_(0, 0.1)

    def forward(self, state):
        'In its method forward,the state is passed into the neural network and finally the softmax to get the output'
        
        state=torch.tensor(state,dtype=torch.float).unsqueeze(0)
        state = self.layerInput(state)
        state = Functions.relu(state)
        state = self.HiddenLayer(state)
        state= Functions.relu(state)
        state = self.layerOutput(state)
        actions_Prob = Functions.softmax(state, dim=1)
        return actions_Prob


class Criticnetwork(nn.Module):
    'Critics NN is such that the input,hidden_layersize,output has been defined in the constructer'
    def __init__(self):
       
        super(Criticnetwork, self).__init__()
        self.layerCInput = nn.Linear(4, 64)
        self.layerCInput.weight.data.normal_(0, 0.1)
        self.HiddenLayer = nn.Linear(64, 64)
        self.HiddenLayer.weight.data.normal_(0, 0.1)
        self.layerCOutput = nn.Linear(64, 1)
        self.layerCOutput.weight.data.normal_(0, 0.1)

    def forward(self, state):
        'In the forward method forward of this class,the state is passed into the neural network to get the value of that state'
        state=torch.tensor(state,dtype=torch.float).unsqueeze(0)
        state= self.layerCInput(state)
        state = F.relu(state)
        state_value = self.layerCOutput(state)
        return state_value


class Actor(object):
    'The input,output for the NN and optimizer are intialized in this class constructer,'
    'The actors NN  also has been called in the present class constructer '
    def __init__(self, stateSize, n_action, LR_Actor):
        self.stateSize = stateSize
        self.n_action = n_action
        self.LR_Actor = LR_Actor
        self.ActorNet = Actornetwork(n_action)
        self.optimizerActor = torch.optim.Adam(self.ActorNet.parameters(), lr=self.LR_Actor)

    def learn(self, state, action, td):
        'The actor takes in state,action,td (obtained from critic)to optimize on the loss value'
        act_prob = self.ActorNet.forward(state)
        dist=torch.distributions.Categorical(act_prob)
        log_probs = dist.log_prob(action)
        loss_actor = - log_probs * td
        
        
        self.optimizerActor.zero_grad()
        loss_actor.backward()
        self.optimizerActor.step()

    def choose_action(self, state):
        'This method lets the agent to choose only one action from the different set of actions'
        actions_Prob_Vector = self.ActorNet.forward(state)
        action = torch.multinomial(actions_Prob_Vector,num_samples=1)
        return action


class Critic(object):
    'LR,the loss functions of the critic,and the optimizer of the critic has been defined in its constructer '
        
    def __init__(self, LR_Critic):
        
        self.LR_Critic = LR_Critic
        self.CriticNet = Criticnetwork()
        self.CriticLoss = nn.MSELoss()
        self.optimizerCritic = torch.optim.Adam(self.CriticNet.parameters(), lr=self.LR_Critic)

    def learn(self, state, reward, next_state, gamma):
        'The critic evalutes on one step td error,the td_error is the loss function for the critic'
       
        value_state = self.CriticNet.forward(state)
        value_state_ = self.CriticNet.forward(next_state)
        one_step_td_error = reward + gamma * value_state_ - value_state
        loss_critic = one_step_td_error
        self.optimizerCritic.zero_grad()
        loss_critic.backward()
        self.optimizerCritic.step()
        return one_step_td_error.detach()
    

           