import numpy as np
import scipy.sparse as sp
from common.BatchEncoder import BatchEncoder 
from common.SparseBatchEncoder import SparseBatchEncoder
from common.SparseBatchTransformer import SparseBatchTransformer
from common.BatchTransformer import BatchTransformer
from common.ValueTransformer import ValueTransformer
from common.SequenceEncoder import SequenceEncoder

class DataGenerator :
    
    def __init__(self, data_ids, sources, batch_size=32, inputs=None, outputs=None, randomizers=[], shuffle=True, densify_batch_matrices=False, move_outputs_to_inputs=False) :
        if move_outputs_to_inputs :
            inputs.extend(outputs)
            outputs = [
                {
                    'id' : 'dummy_output',
                    'source_type' : 'zeros',
                    'dim' : (1,),
                    'sparsify' : False
                }
            ]

        self.data_ids = data_ids
        self.sources = sources
        self.batch_size = batch_size
        self.inputs = inputs
        self.outputs = outputs
        self.randomizers = randomizers
        self.shuffle = shuffle
        self.densify_batch_matrices = densify_batch_matrices
        
        self._init_encoders()

        if isinstance(self.shuffle, DataGenerator) :
            print("shuffle is Datagenerator")
            self.indexes = self.shuffle.indexes
        else :
            self.indexes = np.arange(len(self.data_ids))

        self.on_epoch_end()
    
    def _init_encoders(self) :
        self.encoders = {}
        self.transformers = {}
        
        for input_dict in self.inputs :
            if 'sparse' not in input_dict or not input_dict['sparse'] :
                if 'encoder' in input_dict and input_dict['encoder'] is not None and isinstance(input_dict['encoder'], SequenceEncoder) :
                    input_dict['encoder'] = BatchEncoder(input_dict['encoder'])
                elif 'transformer' in input_dict and input_dict['transformer'] is not None and isinstance(input_dict['transformer'], ValueTransformer) :
                    input_dict['transformer'] = BatchTransformer(input_dict['transformer'])
            else :
                sparse_mode = 'row'
                if 'sparse_mode' in input_dict :
                    sparse_mode = input_dict['sparse_mode']
                if 'encoder' in input_dict and input_dict['encoder'] is not None and isinstance(input_dict['encoder'], SequenceEncoder) :
                    input_dict['encoder'] = SparseBatchEncoder(input_dict['encoder'], sparse_mode=sparse_mode)
                elif 'transformer' in input_dict and input_dict['transformer'] is not None and isinstance(input_dict['transformer'], ValueTransformer) :
                    input_dict['transformer'] = SparseBatchTransformer(input_dict['transformer'], sparse_mode=sparse_mode)
            
            if 'encoder' in input_dict and input_dict['encoder'] is not None :
                self.encoders[input_dict['id']] = input_dict['encoder']
            elif 'transformer' in input_dict and input_dict['transformer'] is not None :
                self.transformers[input_dict['id']] = input_dict['transformer']
        
            if 'encoder' not in input_dict :
                input_dict['encoder'] = None
            if 'transformer' not in input_dict :
                input_dict['transformer'] = None
        if self.outputs is not None :
            for output_dict in self.outputs :
                if output_dict['source_type'] != 'zeros' and ('transformer' in output_dict and output_dict['transformer'] is not None) :
                    if ('sparse' not in output_dict or not output_dict['sparse']) and isinstance(output_dict['transformer'], ValueTransformer) :
                        output_dict['transformer'] = BatchTransformer(output_dict['transformer'])
                    elif isinstance(output_dict['transformer'], ValueTransformer) :
                        sparse_mode = 'row'
                        if 'sparse_mode' in output_dict :
                            sparse_mode = output_dict['sparse_mode']
                        output_dict['transformer'] = SparseBatchTransformer(output_dict['transformer'], sparse_mode=sparse_mode)

                    self.transformers[output_dict['id']] = output_dict['transformer']
                else :
                    self.transformers[output_dict['id']] = None

    def __len__(self) :
        return int(np.floor(len(self.data_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate random samples for current batch
        for randomizer in self.randomizers :
            randomizer.generate_random_sample(self.batch_size, self.data_ids[indexes])


        # Generate data
        batch_tuple = self._generate_batch(self.data_ids[indexes])

        return batch_tuple

    def on_epoch_end(self) :
        if not isinstance(self.shuffle, DataGenerator) and self.shuffle == True :
            np.random.shuffle(self.indexes)
        
        for randomizer in self.randomizers :
           randomizer.generate_random_sample(self.batch_size)
    
    def _generate_batch(self, batch_indexes) :
        
        generated_inputs = []
        generated_outputs = None
        
        #Generate inputs
        for input_dict in self.inputs :
            input_source = self.sources[input_dict['source']]
            if input_dict['source_type'] == 'dataframe' :
               
                source_input = input_source.iloc[batch_indexes]
                generated_input = []
                i = 0
                for _, row in source_input.iterrows() :
                    generated_input.append(input_dict['extractor'](row, i))
                    i += 1
                
                
                if input_dict['encoder'] is not None :
                    if isinstance(input_dict['encoder'], (BatchEncoder, SparseBatchEncoder)) :
                        generated_input = input_dict['encoder'](generated_input)
                    else :
                        generated_input = np.concatenate([np.expand_dims(input_dict['encoder'](inp), axis=0) for inp in generated_input], axis=0)
                elif input_dict['transformer'] is not None :
                    if isinstance(input_dict['transformer'], (BatchTransformer, SparseBatchTransformer)) :
                        generated_input = input_dict['transformer'](np.vstack(generated_input))
                    else :
                        generated_input = np.concatenate([np.expand_dims(input_dict['transformer'](inp), axis=0) for inp in generated_input], axis=0)
                else :
                    generated_input = np.vstack(generated_input)
                
                if 'dim' in input_dict :
                    new_dim = tuple([self.batch_size] + list(input_dict['dim']))
                    generated_input = np.reshape(generated_input, new_dim)
                
                generated_inputs.append(generated_input)
            elif input_dict['source_type'] == 'matrix' :
                generated_input = input_source[batch_indexes]
                if self.densify_batch_matrices and isinstance(input_source, (sp.csr_matrix, sp.csc_matrix)) :
                    generated_input = np.array(generated_input.todense())
                
                if input_dict['extractor'] is not None :
                    generated_input = np.vstack([input_dict['extractor'](generated_input[i], i) for i in range(0, generated_input.shape[0])])

                if input_dict['transformer'] is not None :
                    if isinstance(input_dict['transformer'], (BatchTransformer, SparseBatchTransformer)) :
                        generated_input = input_dict['transformer'](generated_input)
                    else :
                        generated_input = np.concatenate([np.expand_dims(input_dict['transformer'](generated_input[i]), axis=0) for i in range(generated_input.shape[0])], axis=0)
                
                if 'dim' in input_dict :
                    new_dim = tuple([self.batch_size] + list(input_dict['dim']))
                    generated_input = np.reshape(generated_input, new_dim)
                
                generated_inputs.append(generated_input)
            else :
                raise NotImplementedError()
        
        #Generate outputs
        if self.outputs is not None :
            generated_outputs = []
            for output_dict in self.outputs :
                if output_dict['source_type'] == 'matrix' :
                    output_source = self.sources[output_dict['source']]
                    
                    generated_output = output_source[batch_indexes]
                    if self.densify_batch_matrices and isinstance(output_source, (sp.csr_matrix, sp.csc_matrix)) :
                        generated_output = np.array(generated_output.todense())
                    
                    if output_dict['extractor'] is not None :
                        generated_output = np.vstack([output_dict['extractor'](generated_output[i], i) for i in range(0, generated_output.shape[0])])

                    if output_dict['transformer'] is not None :
                        if isinstance(output_dict['transformer'], (BatchTransformer, SparseBatchTransformer)) :
                            generated_output = output_dict['transformer'](generated_output)
                        else :
                            generated_output = np.concatenate([np.expand_dims(output_dict['transformer'](generated_output[i]), axis=0) for i in range(generated_output.shape[0])], axis=0)

                    if 'dim' in output_dict :
                        new_dim = tuple([self.batch_size] + list(output_dict['dim']))
                        generated_output = np.reshape(generated_output, new_dim)
                    
                    generated_outputs.append(generated_output)
                elif output_dict['source_type'] == 'dataframe' :
                    output_source = self.sources[output_dict['source']]
                    
                    source_output = output_source.iloc[batch_indexes]
                    generated_output = []
                    i = 0
                    for _, row in source_output.iterrows() :
                        generated_output.append(output_dict['extractor'](row, i))
                        i += 1
                    
                
                    if output_dict['transformer'] is not None :
                        if isinstance(output_dict['transformer'], (BatchTransformer, SparseBatchTransformer)) :
                            generated_output = output_dict['transformer'](np.vstack(generated_output))
                        else :
                            generated_output = np.concatenate([np.expand_dims(output_dict['transformer'](inp), axis=0) for inp in generated_output], axis=0)
                    else :
                        generated_output = np.vstack(generated_output)

                    if 'dim' in output_dict :
                        new_dim = tuple([self.batch_size] + list(output_dict['dim']))
                        generated_output = np.reshape(generated_output, new_dim)
                    
                    generated_outputs.append(generated_output)
                elif output_dict['source_type'] == 'zeros' :
                    if 'dim' in output_dict :
                        new_dim = tuple([self.batch_size] + list(output_dict['dim']))
                        generated_outputs.append(np.zeros(new_dim))
                    else :
                        generated_outputs.append(np.zeros(self.batch_size))
                else :
                    raise NotImplementedError()

        if generated_outputs is not None :
            return generated_inputs, generated_outputs
        
        return generated_inputs
