from __future__ import print_function

import numpy as np

from common.CountExtractor import CountExtractor
from common.KerasSequenceDataGenerator import KerasSequenceDataGenerator
from common.SequenceExtractor import SequenceExtractor
from common.OneHotEncoder import OneHotEncoder
from common.CategoricalEncoder import CategoricalEncoder
from neuralNetwork.KerasModelHelper import *

def prepareData(data, valid_set_size=0.025, test_set_size=0.025, canonical_pas=False, no_dse_canonical_pas=False, sampleSize=None):
    
    plasmid_df = data['plasmid_df']
    plasmid_cuts = data['plasmid_cuts']
    
    if sampleSize is not None:        
        keep_index = plasmid_df.sample(n=sampleSize).index       
        plasmid_df = plasmid_df.iloc[keep_index].copy()
        plasmid_cuts = plasmid_cuts[keep_index, :]

    if canonical_pas :
        keep_index = np.nonzero(plasmid_df.seq.str.slice(70, 76) == 'AATAAA')[0]
        plasmid_df = plasmid_df.iloc[keep_index].copy()
        plasmid_cuts = plasmid_cuts[keep_index, :]

    if no_dse_canonical_pas :
        keep_index = np.nonzero(~plasmid_df.seq.str.slice(76).str.contains('AATAAA'))[0]
        plasmid_df = plasmid_df.iloc[keep_index].copy()
        plasmid_cuts = plasmid_cuts[keep_index, :]
    
    #Generate training and test set indexes
    plasmid_index = np.arange(len(plasmid_df), dtype=np.int)

    plasmid_train_index = plasmid_index[:-int(len(plasmid_df) * (valid_set_size + test_set_size))]
    plasmid_valid_index = plasmid_index[plasmid_train_index.shape[0]:-int(len(plasmid_df) * test_set_size)]
    plasmid_test_index = plasmid_index[plasmid_train_index.shape[0] + plasmid_valid_index.shape[0]:]

    print('Set size = ' + str(plasmid_index.shape[0]))
    print('Training set size = ' + str(plasmid_train_index.shape[0]))
    print('Validation set size = ' + str(plasmid_valid_index.shape[0]))
    print('Test set size = ' + str(plasmid_test_index.shape[0]))
    
    return plasmid_df, plasmid_cuts, plasmid_index, plasmid_train_index, plasmid_valid_index, plasmid_test_index

def load_data_training(data, batch_size=32, valid_set_size=0.025, test_set_size=0.025, canonical_pas=False, no_dse_canonical_pas=False, sampleSize=None):

    plasmid_df, plasmid_cuts, plasmid_index, plasmid_train_index, plasmid_valid_index, plasmid_test_index = prepareData(data, valid_set_size=valid_set_size, test_set_size=test_set_size, canonical_pas=canonical_pas, no_dse_canonical_pas=no_dse_canonical_pas, sampleSize=sampleSize)
    unique_libraries = 13

    pos_shifter = get_bellcurve_shifter()

    plasmid_training_gens = {
        gen_id : KerasSequenceDataGenerator(
            idx,
            {'df' : plasmid_df, 'cuts' : plasmid_cuts},
            batch_size=batch_size,
            inputs = [
                {
                    'id' : 'seq',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : SequenceExtractor('padded_seq', start_pos=180, end_pos=180 + 205, shifter=pos_shifter if gen_id == 'train' else None),
                    'encoder' : OneHotEncoder(seq_length=205),
                    'dim' : (205, 4, 1),
                    'sparsify' : False
                },
                {
                    'id' : 'lib',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['library'],
                    'encoder' : CategoricalEncoder(n_categories=len(unique_libraries), categories=unique_libraries),
                    'sparsify' : False
                },
                {
                    'id' : 'distal_pas',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: 1 if row['library_index'] in [2, 5, 8, 11, 20, 40] else 0,
                    'encoder' : None,
                    'sparsify' : False
                },
                {
                    'id' : 'total_count',
                    'source_type' : 'matrix',
                    'source' : 'cuts',
                    'extractor' : CountExtractor(start_pos=180, end_pos=180 + 205, static_poses=[-1], shifter=pos_shifter if gen_id == 'train' else None, sparse_source=False),
                    'transformer' : lambda t: np.sum(t),
                    'dim' : (1,),
                    'sparsify' : False
                },
                {
                    'id' : 'prox_usage',
                    'source_type' : 'matrix',
                    'source' : 'cuts',
                    'extractor' : CountExtractor(start_pos=180, end_pos=180 + 205, static_poses=[-1], shifter=pos_shifter if gen_id == 'train' else None, sparse_source=False),
                    'transformer' : lambda t: iso_normalizer(t),
                    'dim' : (1,),
                    'sparsify' : False
                },
                {
                    'id' : 'prox_cuts',
                    'source_type' : 'matrix',
                    'source' : 'cuts',
                    'extractor' : CountExtractor(start_pos=180, end_pos=180 + 205, static_poses=[-1], shifter=pos_shifter if gen_id == 'train' else None, sparse_source=False),
                    'transformer' : lambda t: cut_normalizer(t),
                    'dim' : (206,),
                    'sparsify' : False
                }
            ],
            outputs = [
                {
                    'id' : 'dummy_output',
                    'source_type' : 'zeros',
                    'dim' : (1,),
                    'sparsify' : False
                }
            ],
            randomizers = [pos_shifter] if gen_id == 'train' else [],
            shuffle = True,
            densify_batch_matrices=True
        ) for gen_id, idx in [('all', plasmid_index), ('train', plasmid_train_index), ('valid', plasmid_valid_index), ('test', plasmid_test_index)]
    }
    
    return plasmid_training_gens

def load_data_prediction(data, seq_length, seq_startpos=0, batch_size=32, valid_set_size=0.025, test_set_size=0.025, canonical_pas=False, no_dse_canonical_pas=False, sampleSize=None):
    plasmid_df, plasmid_cuts, plasmid_index, plasmid_train_index, plasmid_valid_index, plasmid_test_index = prepareData(data, valid_set_size=valid_set_size, test_set_size=test_set_size, canonical_pas=canonical_pas, no_dse_canonical_pas=no_dse_canonical_pas, sampleSize=sampleSize)
    unique_libraries = plasmid_df['library'].unique()
    pos_shifter = get_bellcurve_shifter()
    seq_endpos = seq_startpos + seq_length
    seq_dim = seq_length + 1
    
    plasmid_prediction_gens = {
        gen_id : KerasSequenceDataGenerator(
            idx,
            {'df' : plasmid_df, 'cuts' : plasmid_cuts},
            batch_size=batch_size,
            inputs = [
                {
                    'id' : 'seq',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : SequenceExtractor('padded_seq', start_pos=seq_startpos, end_pos=seq_endpos),
                    'encoder' : OneHotEncoder(seq_length=seq_length),
                    'dim' : (seq_length, 4, 1),
                    'sparsify' : False
                },
                {
                    'id' : 'lib',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['library'],
                    'encoder' : CategoricalEncoder(n_categories=len(unique_libraries), categories=unique_libraries),
                    'sparsify' : False
                },
                {
                    'id' : 'distal_pas',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: 1 if row['library_index'] in [2, 5, 8, 11, 20, 40] else 0,
                    'encoder' : None,
                    'sparsify' : False
                }
            ],
            outputs = [
                {
                    'id' : 'prox_usage',
                    'source_type' : 'matrix',
                    'source' : 'cuts',
                    'extractor' : CountExtractor(start_pos=180, end_pos=seq_endpos, static_poses=[-1], sparse_source=False),
                    'transformer' : lambda t: iso_normalizer(t),
                    'dim' : (1,),
                    'sparsify' : False
                },
                {
                    'id' : 'prox_cuts',
                    'source_type' : 'matrix',
                    'source' : 'cuts',
                    'extractor' : CountExtractor(start_pos=180, end_pos=seq_endpos, static_poses=[-1], sparse_source=False),
                    'transformer' : lambda t: cut_normalizer(t),
                    'dim' : (seq_dim,),
                    'sparsify' : False
                }
            ],
            randomizers = [pos_shifter] if gen_id == 'train' else [],
            shuffle = False,
            densify_batch_matrices=True
        ) for gen_id, idx in [('all', plasmid_index), ('train', plasmid_train_index), ('valid', plasmid_valid_index), ('test', plasmid_test_index)]
    }

    return plasmid_prediction_gens