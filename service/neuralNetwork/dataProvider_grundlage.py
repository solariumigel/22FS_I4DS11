import scipy.io as spio
import pandas as pd 

def load(full_file_path) :
	file_path, file_name_prefix = '/'.join(full_file_path.split('/')[:-1]), full_file_path.split('/')[-1]

	data_dict = {}
	with open(file_path + '/' + file_name_prefix + "_fileindex.txt", 'rt') as f :
		for line in f.readlines() :
			file_name, file_name_suffix, file_type = line.strip().split('\t')

			if file_type == 'csv' :
				data = pd.read_csv(file_path + '/' + file_name, sep='\t')
			elif file_type == 'mat' :
				data = spio.loadmat(file_path + '/' + file_name)['data_mat']

			data_dict[file_name_suffix] = data

	return data_dict