from __future__ import print_function

import numpy as np

from common.KerasSequenceDataGenerator import KerasSequenceDataGenerator
from common.SequenceExtractor import SequenceExtractor
from common.OneHotEncoder import OneHotEncoder
from common.CategoricalEncoder import CategoricalEncoder
from neuralNetwork.KerasModelHelper import *

def prepareData(data, valid_set_size=0.025, test_set_size=0.025, canonical_pas=False, no_dse_canonical_pas=False, sampleSize=None):
    
    if sampleSize is not None:        
        keep_index = data.sample(n=sampleSize).index       
        data = data.iloc[keep_index].copy()
    
    if canonical_pas :
        keep_index = np.nonzero(data.seq.str.slice(70, 76) == 'AATAAA')[0]
        data = data.iloc[keep_index].copy()

    if no_dse_canonical_pas :
        keep_index = np.nonzero(~data.seq.str.slice(76).str.contains('AATAAA'))[0]
        data = data.iloc[keep_index].copy()
    
    #Generate training and test set indexes
    plasmid_index = np.arange(len(data), dtype=np.int)

    plasmid_train_index = plasmid_index[:-int(len(data) * (valid_set_size + test_set_size))]
    plasmid_valid_index = plasmid_index[plasmid_train_index.shape[0]:-int(len(data) * test_set_size)]
    plasmid_test_index = plasmid_index[plasmid_train_index.shape[0] + plasmid_valid_index.shape[0]:]

    print('Set size = ' + str(plasmid_index.shape[0]))
    print('Training set size = ' + str(plasmid_train_index.shape[0]))
    print('Validation set size = ' + str(plasmid_valid_index.shape[0]))
    print('Test set size = ' + str(plasmid_test_index.shape[0]))
    
    return data, plasmid_index, plasmid_train_index, plasmid_valid_index, plasmid_test_index

def load_data_prediction(data, seq_length, seq_startpos=0, batch_size=32, valid_set_size=0.025, test_set_size=0.025, canonical_pas=False, no_dse_canonical_pas=False, sampleSize=None):
    data, plasmid_index, plasmid_train_index, plasmid_valid_index, plasmid_test_index = prepareData(data, valid_set_size=valid_set_size, test_set_size=test_set_size, canonical_pas=canonical_pas, no_dse_canonical_pas=no_dse_canonical_pas, sampleSize=sampleSize)
    seq_endpos = seq_startpos + seq_length

    pos_shifter = get_bellcurve_shifter()

    plasmid_prediction_gens = {
        gen_id : KerasSequenceDataGenerator(
            idx,
            {'df' : data},
            batch_size=batch_size,
            inputs = [
                {
                    'id' : 'seq',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : SequenceExtractor('sequence', start_pos=seq_startpos, end_pos=seq_endpos),
                    'encoder' : OneHotEncoder(seq_length=seq_length),
                    'dim' : (seq_length, 4, 1),
                    'sparsify' : False
                },
                {
                    'id' : 'distal_pas',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: 1 if row['direction'] == '+' else 0,
                    'encoder' : None,
                    'sparsify' : False
                }
            ],
            outputs = [
                {
                    'id' : 'prox_usage',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['frequency'],
                    'transformer' : lambda t: np.sum(t),
                    'dim' : (1,),
                    'sparsify' : False
                },
            ],
            randomizers = [pos_shifter] if gen_id == 'train' else [],
            shuffle = False,
            densify_batch_matrices=True
        ) for gen_id, idx in [('all', plasmid_index), ('train', plasmid_train_index), ('valid', plasmid_valid_index), ('test', plasmid_test_index)]
    }

    return plasmid_prediction_gens

def load_data_prediction_withLibrary(data, seq_length, seq_startpos=0, batch_size=32, valid_set_size=0.025, test_set_size=0.025, canonical_pas=False, no_dse_canonical_pas=False, sampleSize=None):
    data, plasmid_index, plasmid_train_index, plasmid_valid_index, plasmid_test_index = prepareData(data, valid_set_size=valid_set_size, test_set_size=test_set_size, canonical_pas=canonical_pas, no_dse_canonical_pas=no_dse_canonical_pas, sampleSize=sampleSize)
    unique_libraries = data['library'].unique()
    seq_endpos = seq_startpos + seq_length

    pos_shifter = get_bellcurve_shifter()

    plasmid_prediction_gens = {
        gen_id : KerasSequenceDataGenerator(
            idx,
            {'df' : data},
            batch_size=batch_size,
            inputs = [
                {
                    'id' : 'seq',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : SequenceExtractor('sequence', start_pos=seq_startpos, end_pos=seq_endpos),
                    'encoder' : OneHotEncoder(seq_length=seq_length),
                    'dim' : (seq_length, 4, 1),
                    'sparsify' : False
                },
                {
                    'id' : 'lib',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['direction'],
                    'encoder' : CategoricalEncoder(n_categories=len(unique_libraries), categories=unique_libraries),
                    'sparsify' : False
                },
                {
                    'id' : 'distal_pas',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: 1 if row['direction'] == '+' else 0,
                    'encoder' : None,
                    'sparsify' : False
                }
            ],
            outputs = [
                {
                    'id' : 'prox_usage',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['frequency'],
                    'transformer' : lambda t: np.sum(t),
                    'dim' : (1,),
                    'sparsify' : False
                },
            ],
            randomizers = [pos_shifter] if gen_id == 'train' else [],
            shuffle = False,
            densify_batch_matrices=True
        ) for gen_id, idx in [('all', plasmid_index), ('train', plasmid_train_index), ('valid', plasmid_valid_index), ('test', plasmid_test_index)]
    }

    return plasmid_prediction_gens