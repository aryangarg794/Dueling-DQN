from sortedcontainers import SortedList
from typing import Self

class ValueSortedDict:
    def __init__(self: Self):
        self.data = {}
        self.sorted_values = SortedList()

    def __setitem__(self: Self, key: int, value: float | int):
        if key in self.data:
            self.sorted_values.remove((self.data[key], key))
        self.data[key] = value
        self.sorted_values.add((value, key))

    def get(self: Self, key: int):
        return self.data.get(key)
    
    def get_rank(self: Self, key: int): 
        if key in self.data:
            return self.sorted_values.index((self.data[key], key))
        
    def get_by_rank(self: Self, rank: int): 
        return self.sorted_values[rank]

    def get_sorted_items(self: Self):
        return [(key, value) for value, key in self.sorted_values]

    def __repr__(self: Self): 
        return f'{self.get_sorted_items()}' 