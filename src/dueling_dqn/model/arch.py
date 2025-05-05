import gymnasium as gym
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from copy import deepcopy
from torch import Tensor
from typing import Self, List, Tuple

from dueling_dqn.experience_replay.per import RankBasedReplayBuffer, ProportionalReplayBuffer, BasicBuffer

class FeatureExtractorAtari(nn.Module):
    
    def __init__(
        self: Self, 
        in_channels: int = 4,
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
    
    def __getitem__(self: Self, index: int) -> nn.Module:
        return self.layers[index]


class DuelingArch(nn.Module):
    
    def __init__(
        self: Self, 
        env: gym.Env, 
        hidden_layers_extractor: List = list([128, 512, 64]), 
        start_epsilon: float = 1,
        max_decay: float = 0.1,
        decay_steps: int = 1000,
        atari: bool = False,
        device: str = 'cpu',
        *args, 
        **kwargs
    ) -> None:
        super(DuelingArch, self).__init__(*args, **kwargs)

        self.env = env
        self.num_actions = env.action_space.n
        self.start_epsilon = start_epsilon
        self.epsilon = start_epsilon
        self.max_decay = max_decay
        self.decay_steps = decay_steps
        self.device = device
        
        if atari: 
            self.initial_extractor = FeatureExtractorAtari(out_dim=hidden_layers_extractor[-1], *args, *kwargs)
        else:
            self.initial_extractor = nn.Sequential(
                nn.Linear(np.prod(self.env.observation_space.shape), 
                                               hidden_layers_extractor[0]), 
                nn.ReLU()
            )

            for i in range(1, len(hidden_layers_extractor)):
                self.initial_extractor.append(nn.Linear(hidden_layers_extractor[i-1], hidden_layers_extractor[i])) 
                self.initial_extractor.append(nn.ReLU()) 
        
        self.value_stream = nn.Linear(hidden_layers_extractor[-1], 1)
        self.advantage_stream = nn.Linear(hidden_layers_extractor[-1], self.num_actions)
    
    def forward(self: Self, obs: Tensor) -> Tensor:
        representation = self.initial_extractor(obs)
        values = self.value_stream(representation)
        advantages = self.advantage_stream(representation)
        
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))
    
    def epsilon_greedy(self: Self, obs: np.ndarray, dim: int = -1) -> Tensor:
        rng = np.random.random()
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, *obs.shape)
            if rng < self.epsilon:
                action = self.env.action_space.sample()
                action = torch.tensor(action)
            else:
                q_values = self(obs)
                action = torch.argmax(q_values, dim=dim)
            return action.detach().item()

    def epsilon_decay(self: Self, step: int) -> None:
        self.epsilon = self.max_decay + (self.start_epsilon
                                         - self.max_decay) * max(0, (self.decay_steps - step) / self.decay_steps)    
    
 
class DuelingNetwork:
    
    def __init__(
        self: Self, 
        env: gym.Env, 
        atari: bool = False, 
        lr: float = 6.25e-5,
        buffer_type: 'str' = 'rank', 
        buffer_size: int = int(1e6),
        gamma: float = 0.99,
        tau: float = 0.005, 
        device: str = 'cpu', 
        max_norm: float = 1, 
        stop_anneal: int = 1000, 
        *args, 
        **kwargs
    ) -> None:
        self.device = device 
        
        self.net = DuelingArch(env=env, atari=atari, device=device, **kwargs).to(device=self.device)
        self.target_net = deepcopy(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        # freeze the target model 
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        match buffer_type:
            case 'rank': 
                self.buffer = RankBasedReplayBuffer(
                    capacity=buffer_size, observation_space=env.observation_space, device=self.device, 
                    stop_anneal=stop_anneal
                )
            case 'prop':
                self.buffer = ProportionalReplayBuffer(
                    capacity=buffer_size, observation_space=env.observation_space, device=self.device,
                    stop_anneal=stop_anneal
                )   
            case 'basic':
                self.buffer = BasicBuffer(
                    capacity=buffer_size, observation_space=env.observation_space, device=self.device,
                    stop_anneal=stop_anneal
                )   

        self.buffer_type = buffer_type
        self.val_env = deepcopy(env)
        self.gamma = gamma
        self.tau = tau
        self.max_norm = max_norm
        self.atari = atari
        
    def loss_func(self: Self, preds: Tensor, true: Tensor, weights: Tensor) -> Tensor:
        return torch.mean((true - preds).pow(2) * weights) 
    
    def epsilon_greedy(self: Self, obs: np.ndarray) -> np.ndarray: 
        return self.net.epsilon_greedy(obs)
    
    def decay(self: Self, step: int) -> None:
        self.buffer.anneal_beta(step)
        self.net.epsilon_decay(step)
    
    def update(
        self: Self, 
        batch: Tuple
    ) -> Tuple:
        states, actions, rewards, next_states, dones, weights, idxs = batch
        
        # construct target values
        with torch.no_grad():
            online_actions = self.net(next_states).max(dim=-1, keepdim=True)[1]
            bootstrapped_values = self.target_net(next_states).gather(dim=-1, index=online_actions)

            td_targets = rewards + self.gamma * bootstrapped_values * (1 - dones)
        
        q_values = self.net(states).gather(dim=-1, index=actions)    
        if self.buffer_type == 'rank' or self.buffer_type == 'prop': 
            loss = self.loss_func(q_values, td_targets.detach(), weights.view(*weights.shape, 1))
        else: 
            loss = F.mse_loss(q_values, td_targets.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # scale the gradients 
        for param in self.net.initial_extractor[-1].parameters():
            param.grad *= 1/np.sqrt(2)
                
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.max_norm)
        self.optimizer.step()
        
        transitions = (states, actions, rewards, next_states, dones)
        return self.get_error(transitions), idxs, loss.item()
    
    def get_error(self: Self, transition: Tuple) -> float:
        states, actions, rewards, next_states, dones = transition 
        with torch.no_grad():
            q_values = self.net(states).gather(dim=-1, index=actions)
            action_next = self.net(next_states).max(dim=-1, keepdim=True)[1]
            q_values_next = self.target_net(next_states).gather(dim=-1, index=action_next)
            return (rewards + self.gamma * q_values_next * (1-dones) - q_values).detach()
    
    def soft_update(self: Self) -> None:
        with torch.no_grad():
            for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                target_param.data.copy_((1-self.tau) * target_param.data + self.tau * param.data)
        
    def hard_update(self: Self) -> None:
        self.target_net.load_state_dict(self.net.state_dict())
        
    def evaluate(self: Self, num_evals: int = 10) -> float:
        self.net.eval()
        with torch.no_grad():
            rewards = []
            for _ in range(num_evals):
                obs, _ = self.val_env.reset()
                done = False
                ep_reward = 0
                while not done:
                    action = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, *obs.shape))
                    obs, reward, terminated, truncated, _ = self.val_env.step(action.cpu().argmax().item())
                    ep_reward += reward
                    done = terminated or truncated
                
                rewards.append(ep_reward)
        self.net.train()
        return np.mean(rewards)
    
    def save(
        self: Self,
        game_name: str
    ) -> None:
        torch.save({
            'state_dict' : self.net.state_dict(),
        }, f'models/Dueling_{game_name}.pt')
        
    def load(
        self, 
        path: str
    ) -> None:
        saved_model = torch.load(path, weights_only=True)
        self.net.load_state_dict(saved_model['state_dict'])
   
    def __repr__(self):
        return f'Dueling Agent'