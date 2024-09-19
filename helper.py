import math

class CosDecay(object):
    def __init__(self,max_value,min_value,num_loops):
        self.max_value=max_value
        self.min_value=min_value
        self.num_loops=num_loops
    def get_value(self,i):
        if i<0:
            i=0
        if i>=self.num_loops:
            i=self.num_loops
        value = (math.cos(i * math.pi / self.num_loops) + 1.0) * 0.5
        value = value * (self.max_value - self.min_value) + self.min_value
        return value

class LinearDecay(object):
    def __init__(self,max_value,min_value,num_loops):
        self.max_value = max_value
        self.min_value = min_value
        self.num_loops = num_loops
    def get_value(self,i):
        if i < 0:
            i = 0
        if i >= self.num_loops:
            i = self.num_loops - 1

        value = (self.max_value - self.min_value) / self.num_loops
        value = i * (-value)
        return value