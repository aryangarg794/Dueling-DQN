import numpy as np

from typing import Self, List, Tuple

class SumTree:
    """Implementation of a sum tree. 
    """
    
    def __init__(
        self: Self,
        size: int 
    ) -> None:
        self.tree = np.zeros((2*size - 1))
        
        self.size = size
        self.pointer = 0 
        self.n_samples = 0 
        
    def update(self: Self, indices: List | np.ndarray | int, td_errors: List | np.ndarray | int) -> None: 
        """Change existing tuples
        """

        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        if not isinstance(td_errors, np.ndarray):
            td_errors = np.array(td_errors)
        
        indices = indices + self.size - 1
        for i, node in enumerate(indices): 
            parent = (node-1) // 2
            change = td_errors[i] - self.tree[node]
            
            while parent >= 0:
                self.tree[parent] += change
                parent = (parent - 1) // 2
                
            
            self.tree[node] = td_errors[i]    

    def add(self: Self, td_errors: List | np.ndarray) -> None: 
        """Add new tuple(s)
        """
        assert len(td_errors) <= self.size
        
        # add new sample 
        if not isinstance(td_errors, np.ndarray):
            td_errors = np.array(td_errors)
        
        idxs = ((np.arange(0, td_errors.shape[0]) + self.pointer) % self.size)
        self.update(idxs, td_errors)  
        
        self.pointer = (self.pointer + td_errors.shape[0]) % self.size
        self.n_samples += td_errors.shape[0]
        
    def sample(self: Self, batch_size: int) -> Tuple[np.ndarray[int], np.ndarray[float]]:
        """Sample based on priorities
        """
        
        assert self.n_samples >= batch_size
        
        p_total = self.total
        n_segments = p_total // batch_size
        intervals = [n_segments * i for i in range(batch_size)] + [p_total]
        sampled_values = np.random.uniform(low=intervals[:-1], high=intervals[1:])
        
        indices = []
        priorities = []
        for sample in sampled_values:
            idx, prio = self._get(sample)
            indices.append(idx)
            priorities.append(prio)
        
        return (np.array(indices), np.array(priorities))
    
    def _get(self: Self, sum: float): 
        node = 0 
        
        while 2 * node + 1 < len(self.tree):
            left = 2 * node + 1
            right = 2 * node + 2

            if sum <= self.tree[left]:
                node = left
            else: 
                sum -= self.tree[left]
                node = right
                
        return node, self.tree[node]
        
        
    @property 
    def total(self: Self) -> float: 
        return self.tree[0]