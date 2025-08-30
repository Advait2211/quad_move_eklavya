import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, n_observations):
        super().__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(n_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
    def forward(self, x):
        return self.critic(x)
    
    def evaluate(self, observations):
        value = self.critic(observations)
        return value
        
        