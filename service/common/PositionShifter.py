import numpy as np

class PositionShifter :
    
    def __init__(self, shift_range, shift_probs) :
        self.shift_range = shift_range
        self.shift_probs = shift_probs
        self.position_shift = 0
        
    def get_random_sample(self, index=None) :
        if index is None :
            return self.position_shift
        else :
            return self.position_shift[index]
    
    def generate_random_sample(self, batch_size=1, batch_indexes=None) :
        self.position_shift = np.random.choice(self.shift_range, size=batch_size, replace=True, p=self.shift_probs)