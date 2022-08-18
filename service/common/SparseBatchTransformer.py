from common.BatchTransformer import BatchTransformer

class SparseBatchTransformer(BatchTransformer) :
    
    def __init__(self, transformer, sparse_mode='row') :
        super(SparseBatchTransformer, self).__init__(transformer)
        
        self.sparse_mode = sparse_mode
    
    def transform(self, values) :
        return self.__call__(values)
    
    def __call__(self, values) :
        if self.sparse_mode == 'row' :
            return self.transform_row_sparse(values)
        elif self.sparse_mode == 'col' :
            return self.transform_col_sparse(values)
        else :
            return self.transform_sparse(values)