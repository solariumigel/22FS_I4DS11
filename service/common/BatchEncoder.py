import numpy as np
import scipy.sparse as sp

class BatchEncoder :
    
    def __init__(self, encoder, memory_efficient=False, memory_efficient_dump_size=30000) :
        self.encoder = encoder
        self.memory_efficient = memory_efficient
        self.memory_efficient_dump_size = memory_efficient_dump_size
    
    def encode(self, seqs) :
        
        batch_dims = tuple([len(seqs)] + list(self.encoder.encode_dims))
        encodings = np.zeros(batch_dims)
        
        self.encode_inplace(seqs, encodings)
        
        return encodings
    
    def encode_inplace(self, seqs, encodings) :
        for i in range(0, len(seqs)) :
            self.encoder.encode_inplace(seqs[i], encodings[i,])
    
    def encode_row_sparse(self, seqs) :
        return sp.csr_matrix(self.encode_sparse(seqs))
    
    def encode_col_sparse(self, seqs) :
        return sp.csc_matrix(self.encode_sparse(seqs))
    
    def encode_sparse(self, seqs) :
        n_cols = np.prod(np.ravel(list(self.encoder.encode_dims)))
        encoding_mat = None

        if not self.memory_efficient or len(seqs) <= self.memory_efficient_dump_size :
            encoding_mat = sp.lil_matrix((len(seqs), n_cols))
            for i in range(0, len(seqs)) :
                self.encoder.encode_inplace_sparse(seqs[i], encoding_mat, i)
        else :
            dump_counter = 0
            dump_max = self.memory_efficient_dump_size
            encoding_acc = None
            encoding_part = sp.lil_matrix((dump_max, n_cols))
            seqs_left = len(seqs)

            for i in range(0, len(seqs)) :
                if dump_counter >= dump_max :
                    if encoding_acc == None :
                        encoding_acc = sp.csr_matrix(encoding_part)
                    else :
                        encoding_acc = sp.vstack([encoding_acc, sp.csr_matrix(encoding_part)])
                    
                    if seqs_left >= dump_max :
                        encoding_part = sp.lil_matrix((dump_max, n_cols))
                    else :
                        encoding_part = sp.lil_matrix((seqs_left, n_cols))

                    dump_counter = 0
                
                dump_counter += 1
                seqs_left -= 1

                self.encoder.encode_inplace_sparse(seqs[i], encoding_part, i % dump_max)

            if encoding_part.shape[0] > 0 :
                encoding_acc = sp.vstack([encoding_acc, sp.csr_matrix(encoding_part)])

            encoding_mat = sp.csr_matrix(encoding_acc)
        
        return encoding_mat
    
    def decode(self, encodings) :
        decodings = []
        for i in range(0, encodings.shape[0]) :
            decodings.append(self.encoder.decode(encodings[i,]))
        
        return decodings
    
    def decode_sparse(self, encoding_mat) :
        decodings = []
        for i in range(0, encoding_mat.shape[0]) :
            decodings.append(self.encoder.decode_sparse(encoding_mat, i))
        
        return decodings
    
    def __call__(self, seqs) :
        return self.encode(seqs)
