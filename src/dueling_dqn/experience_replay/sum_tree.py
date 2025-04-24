import numpy as np

from typing import Self, List

class SumTree:
    """Implementation of a sum tree. 
    """
    
    def __init__(
        self: Self,
        input: List[float] = list([]), 
    ):
        self.list = input
        
        self.construct()
        

    def construct(
        self: Self
    ): 
        """Construct the tree, meant to be ran only once during object creation. 
        """