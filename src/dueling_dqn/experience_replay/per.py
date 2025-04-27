import torch 
import numpy as np 
import gymnasium as gym 

from abc import ABC, abstractmethod
from torch import Tensor
from typing import Self, Tuple, MutableSequence

from dueling_dqn.experience_replay.sum_tree import SumTree

class BufferBase(ABC):
    
    def __init__(
        self: Self, 
        capacity: int, 
        observation_space: gym.Space, 
        action_space: gym.Space, 
        device: str = 'cpu'
    ) -> None:
        self.capacity = capacity
        self.device = device
        
        self.states = torch.empty((capacity, *observation_space.shape), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((capacity, *action_space.shape), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.next_states = torch.empty((capacity, *observation_space.shape), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        
    @abstractmethod
    def sample(
        self: Self,
        batch_size: int
    ) -> Tuple: 
        pass
    
    @abstractmethod
    def add(
        self: Self, 
        transition: Tuple
    ) -> None: 
        pass
    
    def __getitem__(
        self: Self, 
        index: int | Tensor
    ) -> Tuple:
        return (
            self.states[index], 
            self.actions[index],
            self.rewards[index],
            self.next_states[index],
            self.dones[index]
        )
    
    def __setitem__(
        self: Self, 
        index: int | Tensor,
        values: Tuple
    ) -> None:
        self.states[index] = values[0]
        self.actions[index] = values[1]
        self.rewards[index] = values[2]
        self.next_states[index] = values[3]
        self.dones[index] = values[4]
        
class ProportionalReplayBuffer(BufferBase):
    
    def __init__(
        self: Self, 
        *args, 
        **kwargs
    ) -> None:
        super().__init__(**args, **kwargs)
        
        self.tree = SumTree(self.capacity)