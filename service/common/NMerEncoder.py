from common.SequenceEncoder import SequenceEncoder
import numpy as np 

class NMerEncoder(SequenceEncoder) :
    
    def __init__(self, n_mer_len=6, count_n_mers=True) :
        super(NMerEncoder, self).__init__('mer_' + str(n_mer_len), (4**n_mer_len, ))
        
        self.count_n_mers = count_n_mers
        self.n_mer_len = n_mer_len
        self.encode_order = ['A', 'C', 'G', 'T']
        self.n_mers = self._get_ordered_nmers(n_mer_len)
        
        self.encode_map = {
            n_mer : n_mer_index for n_mer_index, n_mer in enumerate(self.n_mers)
        }
        
        self.decode_map = {
            n_mer_index : n_mer for n_mer_index, n_mer in enumerate(self.n_mers)
        }
    
    def _get_ordered_nmers(self, n_mer_len) :
        
        if n_mer_len == 0 :
            return []
        
        if n_mer_len == 1 :
            return list(self.encode_order.copy())
        
        n_mers = []
        
        prev_n_mers = self._get_ordered_nmers(n_mer_len - 1)
        
        for _, prev_n_mer in enumerate(prev_n_mers) :
            for _, nt in enumerate(self.encode_order) :
                n_mers.append(prev_n_mer + nt)
        
        return n_mers
            
    def encode(self, seq) :
        n_mer_vec = np.zeros(self.n_mer_len)
        self.encode_inplace(seq, n_mer_vec)

        return n_mer_vec
    
    def encode_inplace(self, seq, encoding) :
        for i_start in range(0, len(seq) - self.n_mer_len + 1) :
            i_end = i_start + self.n_mer_len
            n_mer = seq[i_start:i_end]
            
            if n_mer in self.encode_map :
                if self.count_n_mers :
                    encoding[self.encode_map[n_mer]] += 1
                else :
                    encoding[self.encode_map[n_mer]] = 1
    
    def encode_inplace_sparse(self, seq, encoding_mat, row_index) :
        for i_start in range(0, len(seq) - self.n_mer_len + 1) :
            i_end = i_start + self.n_mer_len
            n_mer = seq[i_start:i_end]
            
            if n_mer in self.encode_map :
                if self.count_n_mers :
                    encoding_mat[row_index, self.encode_map[n_mer]] += 1
                else :
                    encoding_mat[row_index, self.encode_map[n_mer]] = 1
    
    def decode(self, encoding) :
        n_mers = {}
    
        for i in range(0, encoding.shape[0]) :
            if encoding[i] != 0 :
                n_mers[self.decode_map[i]] = encoding[i]

        return n_mers
    
    def decode_sparse(self, encoding_mat, row_index) :
        encoding = np.ravel(encoding_mat[row_index, :].todense())
        return self.decode(encoding)