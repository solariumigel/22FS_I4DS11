from common.DataGenerator import DataGenerator
from common.NMerEncoder import NMerEncoder
from common.CategoricalEncoder import CategoricalEncoder
import numpy as np

def mask_constant_sequence_regions(df, add_padding=False) :
    if 'mask' in df.columns:
        df['sequence'] = df.apply(map_mask, args={add_padding}, axis=1)

    return df

def map_mask(row, add_padding=False) :
    if row['mask'] is None:
        return ('X' * 186)
    library_mask = row['mask']
    seq_var = ''
    seq = row['sequence']
    for j in range(0, len(seq)) :
        if not add_padding :
            if library_mask[j] == 'N' :
                seq_var += seq[j]
            else :
                seq_var += 'X'
        else :
            if library_mask[j] == 'N' :
                seq_var += seq[j]
            elif j >= 2 and (library_mask[j-1] == 'N' or library_mask[j-2] == 'N') :
                seq_var += seq[j]
            elif j <= len(seq) - 1 - 2 and (library_mask[j+1] == 'N' or library_mask[j+2] == 'N') :
                seq_var += seq[j]
            else :
                seq_var += 'X'
    
    return seq_var

def CreateDataGenerator(data, test_set_size):
    data = mask_constant_sequence_regions(data)

    dataIndex = np.arange(len(data), dtype=np.int)

    train_index = dataIndex[:-int(len(data) * (test_set_size))]
    test_index = dataIndex[train_index.shape[0]:]

    print('Set size = ' + str(dataIndex.shape[0]))
    print('Training set size = ' + str(train_index.shape[0]))
    print('Test set size = ' + str(test_index.shape[0]))

    unique_libaries = data['library_index'].unique()
    hexamer_gens = {
        gen_id : DataGenerator(
            idx,
            {
                'df' : data
            },
            batch_size=len(idx),
            inputs = [
                {
                    'id' : 'use',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['sequence'][:193],
                    'encoder' : NMerEncoder(n_mer_len=6, count_n_mers=True),
                    'sparse' : True,
                    'sparse_mode' : 'col'
                },
                {
                    'id' : 'cse',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['sequence'][197:203],
                    'encoder' : NMerEncoder(n_mer_len=6, count_n_mers=True),
                    'sparse' : True,
                    'sparse_mode' : 'col'
                },
                {
                    'id' : 'dse',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['sequence'][206:246],
                    'encoder' : NMerEncoder(n_mer_len=6, count_n_mers=True),
                    'sparse' : True,
                    'sparse_mode' : 'col'
                },
                {
                    'id' : 'fdse',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['sequence'][246:],
                    'encoder' : NMerEncoder(n_mer_len=6, count_n_mers=True),
                    'sparse' : True,
                    'sparse_mode' : 'col'
                },
                {
                    'id' : 'lib',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['library_index'],
                    'encoder' : CategoricalEncoder(n_categories=len(unique_libaries), categories=unique_libaries),
                    'sparsify' : True
                },
            ],
            outputs = [
                {
                    'id' : 'proximal_usage',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['frequency'],
                    'transformer' : lambda t: t
                }
            ],
            randomizers = [],
            shuffle = False,
        ) for gen_id, idx in [('train', train_index), ('test', test_index)]
    }

    return hexamer_gens

def CreateDataGenerator_WithoutLibrary(data, test_set_size):
    data = mask_constant_sequence_regions(data)

    dataIndex = np.arange(len(data), dtype=np.int)

    train_index = dataIndex[:-int(len(data) * (test_set_size))]
    test_index = dataIndex[train_index.shape[0]:]

    print('Set size = ' + str(dataIndex.shape[0]))
    print('Training set size = ' + str(train_index.shape[0]))
    print('Test set size = ' + str(test_index.shape[0]))

    hexamer_gens = {
        gen_id : DataGenerator(
            idx,
            {
                'df' : data
            },
            batch_size=len(idx),
            inputs = [
                {
                    'id' : 'use',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['sequence'][:193],
                    'encoder' : NMerEncoder(n_mer_len=6, count_n_mers=True),
                    'sparse' : True,
                    'sparse_mode' : 'col'
                },
                {
                    'id' : 'cse',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['sequence'][197:203],
                    'encoder' : NMerEncoder(n_mer_len=6, count_n_mers=True),
                    'sparse' : True,
                    'sparse_mode' : 'col'
                },
                {
                    'id' : 'dse',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['sequence'][206:246],
                    'encoder' : NMerEncoder(n_mer_len=6, count_n_mers=True),
                    'sparse' : True,
                    'sparse_mode' : 'col'
                },
                {
                    'id' : 'fdse',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['sequence'][246:],
                    'encoder' : NMerEncoder(n_mer_len=6, count_n_mers=True),
                    'sparse' : True,
                    'sparse_mode' : 'col'
                }
            ],
            outputs = [
                {
                    'id' : 'proximal_usage',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['frequency'],
                    'transformer' : lambda t: t
                }
            ],
            randomizers = [],
            shuffle = False,
        ) for gen_id, idx in [('train', train_index), ('test', test_index)]
    }

    return hexamer_gens