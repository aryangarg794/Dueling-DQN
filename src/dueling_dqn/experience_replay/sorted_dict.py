from sortedcontainers import SortedList

class ValueSortedDict:
    def __init__(self):
        self.data = {}
        self.sorted_values = SortedList()

    def __setitem__(self, key, value):
        if key in self.data:
            self.sorted_values.remove((self.data[key], key))
        self.data[key] = value
        self.sorted_values.add((value, key))

    def get(self, key):
        return self.data.get(key)
    
    def get_rank(self, key): 
        if key in self.data:
            return self.sorted_values.index((self.data[key], key))
        
    def get_by_rank(self, rank): 
        return self.sorted_values[rank]

    def get_sorted_items(self):
        return [(key, value) for value, key in self.sorted_values]

    def __repr__(self): 
        return f'{self.get_sorted_items()}'