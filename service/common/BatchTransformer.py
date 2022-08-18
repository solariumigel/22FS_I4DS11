import scipy.sparse as sp
import numpy as np

class BatchTransformer :
    
    def __init__(self, transformer) :
        self.transformer = transformer
    
    def transform(self, values) :
        
        batch_dims = tuple([values.shape[0]] + list(self.transformer.transform_dims))
        transforms = np.zeros(batch_dims)
        
        self.transform_inplace(values, transforms)
        
        return transforms
    
    def transform_inplace(self, values, transforms) :
        for i in range(0, values.shape[0]) :
            self.transformer.transform_inplace(values[i], transforms[i,])
    
    def transform_row_sparse(self, values) :
        return sp.csr_matrix(self.transform_sparse(values))
    
    def transform_col_sparse(self, values) :
        return sp.csc_matrix(self.transform_sparse(values))
    
    def transform_sparse(self, values) :
        n_cols = np.prod(np.ravel(list(self.transformer.transform_dims)))
        transform_mat = sp.lil_matrix((values.shape[0], n_cols))
        for i in range(0, values.shape[0]) :
            self.transformer.transform_inplace_sparse(values[i], transform_mat, i)
        
        return transform_mat
    
    def __call__(self, values) :
        return self.transform(values)