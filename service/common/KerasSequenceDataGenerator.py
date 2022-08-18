from common.DataGenerator import DataGenerator
import tensorflow as tf
import numpy as np

class KerasSequenceDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, data_ids, sources, batch_size=32, inputs=None, outputs=None, randomizers=[], shuffle=True, densify_batch_matrices=False, move_outputs_to_inputs=False) :
        self.gen = DataGenerator(data_ids, sources, batch_size=batch_size, inputs=inputs, outputs=outputs, randomizers=randomizers, shuffle=shuffle, densify_batch_matrices=densify_batch_matrices, move_outputs_to_inputs=move_outputs_to_inputs)
        
        self.data_ids = self.gen.data_ids
        self.sources = self.gen.sources
        self.batch_size = self.gen.batch_size
        self.inputs = self.gen.inputs
        self.outputs = self.gen.outputs
        self.randomizers = self.gen.randomizers
        self.shuffle = self.gen.shuffle
        self.densify_batch_matrices = self.gen.densify_batch_matrices
        self.indexes = self.gen.indexes
        self.encoders = self.gen.encoders
        self.transformers = self.gen.transformers

    def __len__(self):
        return self.gen.__len__()

    def __getitem__(self, index):
        return self.gen.__getitem__(index)

    def on_epoch_end(self) :
        self.gen.on_epoch_end()