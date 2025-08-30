import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, n_observations, n_actions, init_noise_std=1.0):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(n_observations, 256), 
            nn.ReLU(), 
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, n_actions), 
            # nn.Softmax(dim=1) Continuoous Action Space
        )
        
        self.std = nn.Parameter(init_noise_std * torch.ones(n_actions))
        self.distribution = None

    def forward(self, x):
        return self.actor(x)
    
    #Creates a Normal distribution based on the outputted actions of the actor network
    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = self.std.expand_as(mean)
        
        self.distribution = Normal(mean, std)
    
    #Chooses a random actions based on the distribution  
    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    #Calculates log probability of actions under the current policy
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    #Action Selection during Evaluation
    def act_inference(self, observations):
        actions_mean = self.forward(observations)
        return actions_mean
    
    
            
