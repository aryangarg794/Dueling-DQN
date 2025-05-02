import torch 
import numpy as np 
import gymnasium as gym 

from abc import ABC, abstractmethod
from sortedcontainers import SortedList
from torch import Tensor
from typing import Self, Tuple, List

from dueling_dqn.experience_replay.sum_tree import SumTree
from dueling_dqn.experience_replay.sorted_dict import ValueSortedDict

class BufferBase(ABC):
    
    def __init__(
        self: Self, 
        capacity: int, 
        observation_space: gym.Space, 
        alpha: float = 0.7,  
        beta: float = 0.5, 
        stop_anneal: int = 100, 
        device: str = 'cpu'
    ) -> None:
        self.capacity = capacity
        self.device = device
        
        self.states = torch.empty((capacity, *observation_space.shape), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((capacity, 1), dtype=torch.int64, device=self.device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.next_states = torch.empty((capacity, *observation_space.shape), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((capacity, 1), dtype=torch.int64, device=self.device)
        
        self.size = 0 
        self.pointer = 0 
        self.epsilon = 1e-5
        self.alpha = alpha
        self.initial_beta = beta
        self.beta = beta
        self.stop_anneal = stop_anneal
        
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
    
    @abstractmethod
    def update(
        self: Self, 
        transition_idxs: List, 
        td_errors: List, 
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
        self.states[index] = torch.as_tensor(values[0], dtype=torch.float32, device=self.device)
        self.actions[index] = torch.as_tensor(values[1], dtype=torch.int64, device=self.device)
        self.rewards[index] = torch.as_tensor(values[2], dtype=torch.float32, device=self.device)
        self.next_states[index] = torch.as_tensor(values[3], dtype=torch.float32, device=self.device)
        self.dones[index] = torch.as_tensor(values[4], dtype=torch.int64, device=self.device)

    def __len__(self: Self) -> int: 
        return self.size
    
    def anneal_beta(self: Self, step: int) -> None:
        if step >= self.stop_anneal:
            return 1.0
        else: 
            self.initial_beta + (1.0 - self.initial_beta) * (step / self.stop_anneal)

class BasicBuffer(BufferBase): 
    
    def __init__(
        self: Self, 
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
    
    def add(
        self: Self, 
        transition: Tuple, 
        error: float
    ) -> None: 
        self[self.pointer] = transition
        
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update(
        self: Self, 
        transition_idxs: List, 
        td_errors: Tensor
    ) -> None: 
        pass
    
class ProportionalReplayBuffer(BufferBase):
    
    def __init__(
        self: Self, 
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.tree = SumTree(self.capacity)
        
    def add(
        self: Self, 
        transition: Tuple, 
        error: float
    ) -> None: 
        self[self.pointer] = transition
        
        self.tree.add([error])
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update(
        self: Self, 
        transition_idxs: List, 
        td_errors: Tensor
    ) -> None: 
        priorities = (np.abs(td_errors.cpu().numpy()) + self.epsilon) ** self.alpha
        self.tree.update(transition_idxs, priorities)
    
    def sample(
        self: Self,
        batch_size: int
    ) -> Tuple: 
        idxs, priorities = self.tree.sample(batch_size)
        
        with torch.no_grad():
            priorities = torch.as_tensor(priorities, dtype=torch.float32, device=self.device) + self.epsilon
            idxs = torch.as_tensor(idxs, dtype=torch.int64, device=self.device)
            
            
            weights = (self.size * priorities).pow(-self.beta)
            max_w = weights.amax()
            weights /= max_w
            states, actions, rewards, next_states, dones = self[idxs]
            return (
                states, 
                actions, 
                rewards, 
                next_states, 
                dones, 
                weights, 
                idxs
            )
            
        
class RankBasedReplayBuffer(BufferBase):
    
    def __init__(
        self: Self, 
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.ranks = ValueSortedDict()

        
    def add(
        self: Self, 
        transition: Tuple, 
        error: float 
    ) -> None: 
        self[self.pointer] = transition
        
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update(
        self: Self, 
        transition_idxs: List, 
        td_errors: Tensor
    ) -> None: 
        for idx, error in zip(transition_idxs, td_errors):
            self.ranks[idx] = np.abs(error)
        
    def sample(
        self: Self,
        batch_size: int
    ) -> Tuple: 
        n_segments = self.size // batch_size
        intervals = [n_segments * i for i in range(batch_size)] + [self.size-1]
        sampled_values = np.random.randint(low=intervals[:-1], high=intervals[1:])
        
        priorities = []
        idxs = []
        for rank in sampled_values:
            _, idx = self.ranks.get_by_rank(rank)
            idxs.append(idx)
            priorities.append(1/(rank+1))
            
        with torch.no_grad():
            priorities = torch.as_tensor(np.array(priorities), dtype=torch.float32, device=self.device) + self.epsilon
            idxs = torch.as_tensor(np.array(idxs), dtype=torch.int64, device=self.device)
            
            weights = (self.size * priorities).pow(-self.beta)
            max_w = weights.amax()
            weights /= max_w
            
            states, actions, rewards, next_states, dones = self[idxs]
            return (
                states, 
                actions, 
                rewards, 
                next_states, 
                dones, 
                weights, 
                idxs
            )
                    