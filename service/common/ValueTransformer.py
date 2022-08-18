class ValueTransformer :
    
    def __init__(self, transformer_type_id, transform_dims) :
        self.transformer_type_id = transformer_type_id
        self.transform_dims = transform_dims
    
    def transform(self, values) :
        raise NotImplementedError()
    
    def transform_inplace(self, values, transform) :
        raise NotImplementedError()
    
    def transform_inplace_sparse(self, values, transform_mat, row_index) :
        raise NotImplementedError()
    
    def __call__(self, values) :
        return self.transform(values)