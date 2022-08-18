import numpy as np
import scipy.sparse as sp
from common.SequenceEncoder import SequenceEncoder 

class OneHotEncoder(SequenceEncoder) :
    
    def __init__(self, seq_length=100, default_fill_value=0) :
        super(OneHotEncoder, self).__init__('one_hot', (seq_length, 4))
        
        self.seq_length = seq_length
        self.default_fill_value = default_fill_value
        self.encode_map = {
            'A' : 0,
            'C' : 1,
            'G' : 2,
            'T' : 3
        }
        self.decode_map = {
                0 : 'A',
                1 : 'C',
                2 : 'G',
                3 : 'T',
                -1 : 'X'
        }
    
    def encode(self, seq) :
        one_hot = np.zeros((self.seq_length, 4))
        self.encode_inplace(seq, one_hot)

        return one_hot
    
    def encode_inplace(self, seq, encoding) :
        for pos, nt in enumerate(list(seq)) :
            if nt in self.encode_map :
                encoding[pos, self.encode_map[nt]] = 1
            elif self.default_fill_value != 0 :
                encoding[pos, :] = self.default_fill_value
    
    def encode_inplace_sparse(self, seq, encoding_mat, row_index) :
        encoding = self.encode(seq)
        encoding_mat[row_index, :] = np.ravel(encoding)
    
    def decode(self, encoding) :
        seq = ''
    
        for pos in range(0, encoding.shape[0]) :
            argmax_nt = np.argmax(encoding[pos, :])
            max_nt = np.max(encoding[pos, :])
            if max_nt == 1 :
                seq += self.decode_map[argmax_nt]
            else :
                seq += self.decode_map[-1]

        return seq
    
    def decode_sparse(self, encoding_mat, row_index) :
        encoding = np.array(encoding_mat[row_index, :].todense()).reshape(-1, 4)
        return self.decode(encoding)

