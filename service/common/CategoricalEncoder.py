from common.SequenceEncoder import SequenceEncoder
import numpy as np

class CategoricalEncoder(SequenceEncoder) :
    
    def __init__(self, n_categories=2, categories=['default_1', 'default_2'], category_index=None) :
        super(CategoricalEncoder, self).__init__('categorical', (n_categories, ))
        
        self.n_categories = n_categories
        self.categories = categories
        self.category_index = category_index
        if self.category_index is None :
            self.category_index = list(np.arange(n_categories, dtype=np.int).tolist())
        
        self.encode_map = {
            category : category_id for category_id, category in zip(self.category_index, self.categories)
        }
        
        self.decode_map = {
            category_id : category for category_id, category in zip(self.category_index, self.categories)
        }
            
    def encode(self, seq) :
        n_mer_vec = np.zeros(self.n_categories)
        self.encode_inplace(seq, n_mer_vec)

        return n_mer_vec
    
    def encode_inplace(self, seq, encoding) :
        encoding[self.encode_map[seq]] = 1
    
    def encode_inplace_sparse(self, seq, encoding_mat, row_index) :
        encoding_mat[row_index, self.encode_map[seq]] = 1
    
    def decode(self, encoding) :
        category = self.decode_map[np.argmax(encoding)]

        return category
    
    def decode_sparse(self, encoding_mat, row_index) :
        encoding = np.ravel(encoding_mat[row_index, :].todense())
        return self.decode(encoding)
