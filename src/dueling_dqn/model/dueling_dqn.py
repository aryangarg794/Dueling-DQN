import gymnasium as gym
import numpy as np
import torch 
import torch.nn as nn 

from torch import Tensor
from typing import Self, List, Tuple

class FeatureExtractorAtari(nn.Module):
    
    def __init__(
        self: Self, 
        in_channels: int = 16,
        hidden_filters: List = list([32, 64, 64]),
        out_dim: int = 64, 
        *args, 
        **kwargs
    ) -> None:
        super(FeatureExtractorAtari, self).__init__(*args, **kwargs)
        
        assert len(hidden_filters) == 3, 'Feature extractor only implements 3 hidden conv layers'
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_filters[0], kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(hidden_filters[0], hidden_filters[1], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(hidden_filters[1], hidden_filters[2], kernel_size=3, stride=1),
            nn.Flatten(start_dim=1),
            nn.Linear(hidden_filters[2] * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
        
    def forward(self, x) -> Tensor:
        x = x / 255.0
        return self.layers(x)


class DuelingNet(nn.Module):
    
    def __init__(
        self: Self, 
        env: gym.Env, 
        hidden_layers_extractor: List = list([512, 256]), 
        out_dim: int = 64, 
        gamma: float = 0.99, 
        start_epsilon: float = 0.99,
        max_decay: float = 0.1,
        decay_steps: int = 1000,
        atari: bool = False,
        *args, 
        **kwargs
    ) -> None:
        super(DuelingNet, self).__init__(*args, **kwargs)

        self.env = env
        self.num_actions = env.action_space.n
        self.start_epsilon = start_epsilon
        self.epsilon = start_epsilon
        self.max_decay = max_decay
        self.decay_steps = decay_steps
        self.gamma = gamma
        
        if atari: 
            self.initial_extractor = FeatureExtractorAtari(out_dim=out_dim, *args, *kwargs)
        else:
            self.initial_extractor = nn.Sequential(
                nn.Linear(np.prod(self.env.observation_space.shape), 
                                               hidden_layers_extractor[0])
            )

            hidden_layers_extractor.append(out_dim)
            for i in range(1, len(hidden_layers_extractor)):
                self.layers.extend(
                    [nn.Linear(hidden_layers_extractor[i-1], hidden_layers_extractor[i]), 
                    nn.ReLU()]
                )
                
        self.value_stream = nn.Linear(out_dim, 1)
        self.advantage_stream = nn.Linear(out_dim, self.num_actions)
    
    def forward(self, obs) -> Tensor:
        representation = self.initial_extractor(obs)
        values = self.value_stream(representation)
        advantages = self.advantage_stream(representation)
        
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))
    
    def epsilon_greedy(self: Self, obs: Tensor, dim: int = -1) -> Tensor:
        rng = np.random.random()
        with torch.no_grad():
            if rng < self.epsilon:
                action = self.env.action_space.sample()
                action = torch.tensor(action)
            else:
                q_values = self(obs)
                action = torch.argmax(q_values, dim=dim)
        return action.detach().cpu().numpy()

    def epsilon_decay(self: Self, step: int) -> None:
        self.epsilon = self.max_decay + (self.start_epsilon
                                         - self.max_decay) * max(0, (self.decay_steps - step) / self.decay_steps)    
    
    def update(
        self: Self, 
        batch: Tuple
    ) -> float:
        return 