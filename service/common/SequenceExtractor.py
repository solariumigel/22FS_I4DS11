class SequenceExtractor :
    
    def __init__(self, df_column, start_pos=0, end_pos=100, shifter=None) :
        self.df_column = df_column
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.shifter = shifter
    
    def extract(self, raw_input, index=None) :
        shift_pos = 0
        if self.shifter is not None :
            shift_pos = self.shifter.get_random_sample(index)
        
        return raw_input[self.df_column][self.start_pos + shift_pos: self.end_pos + shift_pos]
    
    def __call__(self, raw_input, index=None) :
        return self.extract(raw_input, index)
