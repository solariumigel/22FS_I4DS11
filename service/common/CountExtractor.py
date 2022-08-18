import numpy as np 

class CountExtractor :
    
    def __init__(self, df_column=None, start_pos=0, end_pos=100, static_poses=None, shifter=None, sparse_source=False) :
        self.df_column = df_column
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.shifter = shifter
        self.sparse_source = sparse_source
        self.static_poses = static_poses
    
    def extract(self, raw_input, index) :
        shift_pos = 0
        if self.shifter is not None :
            shift_pos = self.shifter.get_random_sample(index)
        
        dense_input = None
        if not self.sparse_source :
            dense_input = raw_input
        else :
            dense_input = np.ravel(raw_input.todense())
        
        if self.df_column is None :
            extracted_values = dense_input[self.start_pos + shift_pos: self.end_pos + shift_pos]
        else :
            extracted_values = dense_input[self.df_column][self.start_pos + shift_pos: self.end_pos + shift_pos]
        
        if self.static_poses is not None :
            if self.df_column is None :
                extracted_values = np.concatenate([extracted_values, dense_input[self.static_poses]], axis=0)
            else :
                extracted_values = np.concatenate([extracted_values, dense_input[self.df_column][self.static_poses]], axis=0)
        
        return extracted_values
    
    def __call__(self, raw_input, index=None) :
        return self.extract(raw_input, index)