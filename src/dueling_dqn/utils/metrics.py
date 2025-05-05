import numpy as np

from collections import deque
from typing import Self, List

class RollingAverage:
    def __init__(
        self: Self, 
        window_size: int = 5, 
    ) -> None:
        
        self.window = deque(maxlen=window_size)
        self.averages = []
        self.num_eps = 0 
        
    def increment_ep(self: Self) -> None:
        self.num_eps += 1

    def update(
        self: Self, 
        value: float
    ) -> None:
        self.window.append(float(value))
        self.averages.append(self.get_average)
        
    @property
    def get_average(self: Self) -> float:
        return sum(self.window) / len(self.window) if self.window else 0.0