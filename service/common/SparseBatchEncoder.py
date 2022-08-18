from common.BatchEncoder import BatchEncoder 

class SparseBatchEncoder(BatchEncoder) :
    
    def __init__(self, encoder, sparse_mode='row') :
        super(SparseBatchEncoder, self).__init__(encoder)
        
        self.sparse_mode = sparse_mode
    
    def encode(self, seqs) :
        return self.__call__(seqs)
    
    def decode(self, encodings) :
        return self.decode_sparse(encodings)
    
    def __call__(self, seqs) :
        if self.sparse_mode == 'row' :
            return self.encode_row_sparse(seqs)
        elif self.sparse_mode == 'col' :
            return self.encode_col_sparse(seqs)
        else :
            return self.encode_sparse(seqs)