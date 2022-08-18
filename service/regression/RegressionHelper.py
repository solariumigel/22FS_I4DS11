import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.optimize as spopt
import os
from regression.datenProvider import CreateDataGenerator
from common.PlotHelper import plotScatterWithAxis

def featureDefinition_withoutLibrary(dataGenerator):
	[X_train_use, X_train_cse, X_train_dse, X_train_fdse], y_train = dataGenerator['train'][0]
	y_train = y_train[0]

	[X_test_use, X_test_cse, X_test_dse, X_test_fdse], y_test = dataGenerator['test'][0]
	y_test = y_test[0]

	X_train = sp.csc_matrix(sp.hstack([X_train_use, X_train_cse, X_train_dse, X_train_fdse]))
	X_test = sp.csc_matrix(sp.hstack([X_test_use, X_test_cse, X_test_dse, X_test_fdse]))

	return X_train, y_train, X_test, y_test

def featureDefinition(dataGenerator):
	[X_train_use, X_train_cse, X_train_dse, X_train_fdse, X_train_lib], y_train = dataGenerator['train'][0]
	y_train = y_train[0]

	[X_test_use, X_test_cse, X_test_dse, X_test_fdse, X_test_lib], y_test = dataGenerator['test'][0]
	y_test = y_test[0]

	#Concatenate hexamer count matrices
	X_train = sp.csc_matrix(sp.hstack([X_train_lib, X_train_use, X_train_cse, X_train_dse, X_train_fdse]))
	X_test = sp.csc_matrix(sp.hstack([X_test_lib, X_test_use, X_test_cse, X_test_dse, X_test_fdse]))

	return X_train, y_train, X_test, y_test

def TrainAndPredict(data, test_set_size, outputPath, featureDefinitionFunction, CreateDataGeneratorFunction, model_version='', data_version='', sampleSize=None):
	if(sampleSize):
		print('Samplesize: ' + str(sampleSize))
		keep_index = data.sample(n=sampleSize).index       
		data = data.iloc[keep_index].copy()

	if not os.path.isdir(outputPath):
		os.makedirs(outputPath)

	dataGenerator = CreateDataGeneratorFunction(data, test_set_size, align_on_cse)

	X_train, y_train, X_test, y_test = featureDefinitionFunction(dataGenerator)

	print("Starting logistic n-mer regression...")

	w_init = np.zeros(X_train.shape[1] + 1)
	lambda_penalty = 0

	(w_bundle, _, _) = spopt.fmin_l_bfgs_b(log_loss, w_init, fprime=log_loss_gradient, args=(X_train, y_train, lambda_penalty), maxiter = 200)

	print("Regression finished.")
	basisFilename = 'regression_' + model_version + '_' + data_version
	weightsFilename = basisFilename + '_weights'
	storeWeights(dataGenerator, w_bundle, outputPath, weightsFilename)

	#Collect weights
	w_0 = w_bundle[0]
	w_L = w_bundle[1:1 + 36]
	w = w_bundle[1 + 36:]

	y_test_pred = get_y_pred(X_test, np.concatenate([w_L, w]), w_0)

	predictionFilename = basisFilename + '_prediction.png'
	plotScatterWithAxis(y_test_pred, y_test, outputPath, predictionFilename, model_version, data_version, 'Prediction')

    #Compute Log Odds values
	keep_index = (y_test < 0.99999)
	y_test_valid = y_test[keep_index]
	y_test_pred_valid = y_test_pred[keep_index]

	logodds_test = np.ravel(safe_log(y_test_valid / (1. - y_test_valid)))
	logodds_test_pred = np.ravel(safe_log(y_test_pred_valid / (1. - y_test_pred_valid)))

	predictionFilename = basisFilename + '_predictionLogodds.png'
	plotScatterWithAxis(logodds_test_pred, logodds_test, outputPath, predictionFilename, model_version, data_version, 'Log Odds values')

def align_on_cse(df) :
	cano_pas1 = 'AATAAA'
	cano_pas2 = 'ATTAAA'

	pas_mutex1_1 = {}
	pas_mutex1_2 = {}

	pas_mutex2_1 = {}

	for pos in range(0, 6) :
		for base in ['A', 'C', 'G', 'T'] :
			if cano_pas1[:pos] + base + cano_pas1[pos+1:] not in pas_mutex1_1 :
				pas_mutex1_1[cano_pas1[:pos] + base + cano_pas1[pos+1:]] = True
			if cano_pas2[:pos] + base + cano_pas2[pos+1:] not in pas_mutex1_2 :
				pas_mutex1_2[cano_pas2[:pos] + base + cano_pas2[pos+1:]] = True

	for pos1 in range(0, 6) :
		for pos2 in range(pos1 + 1, 6) :
			for base1 in ['A', 'C', 'G', 'T'] :
				for base2 in ['A', 'C', 'G', 'T'] :
					if cano_pas1[:pos1] + base1 + cano_pas1[pos1+1:pos2] + base2 + cano_pas1[pos2+1:] not in pas_mutex2_1 :
						pas_mutex2_1[cano_pas1[:pos1] + base1 + cano_pas1[pos1+1:pos2] + base2 + cano_pas1[pos2+1:]] = True

	df['seq_var_aligned'] = df['sequence']
	return df

#Logistic regression prediction
def get_y_pred(X, w, w_0) :
    return 1. / (1. + np.exp(-1. * (X.dot(w) + w_0)))

#Safe log for NLL
def safe_log(x, minval=0.01):
    return np.log(x.clip(min=minval))

#Logistic regression NLL loss
def log_loss(w_bundle, *fun_args) :
    (X, y, lambda_penalty) = fun_args
    w = w_bundle[1:]
    w_0 = w_bundle[0]
    N = float(X.shape[0])

    log_y_zero = safe_log(1. - get_y_pred(X, w, w_0))
    log_y_one = safe_log(get_y_pred(X, w, w_0))

    log_loss = (1. / 2.) * lambda_penalty * np.square(np.linalg.norm(w)) - (1. / N) * np.sum(y * log_y_one + (1. - y) * log_y_zero)

    return log_loss

#Logistic regression NLL gradient
def log_loss_gradient(w_bundle, *fun_args) :
    (X, y, lambda_penalty) = fun_args
    w = w_bundle[1:]
    w_0 = w_bundle[0]
    N = float(X.shape[0])

    y_pred = get_y_pred(X, w, w_0)

    w_0_gradient = - (1. / N) * np.sum(y - y_pred)
    w_gradient = 1. * lambda_penalty * w - (1. / N) * X.T.dot(y - y_pred)

    return np.concatenate([[w_0_gradient], w_gradient])

def storeWeights(dataGenerator, w_bundle, outputPath, fileName):
    w = w_bundle[1 + 36:]

    np.save(os.path.join(outputPath, fileName), w_bundle)

    stored_nmer_weights = {
        'nmer' : [t[1] for t in sorted(dataGenerator['train'].encoders['use'].encoder.decode_map.items(), key=lambda t: t[0])],
        'use' : w[: 4096].tolist(),
        'cse' : w[4096: 2 * 4096].tolist(),
        'dse' : w[2 * 4096: 3 * 4096].tolist(),
        'fdse' : w[3 * 4096: 4 * 4096].tolist(),
    }

    stored_nmer_weights['fdse'] = stored_nmer_weights['fdse'] + [0]*(4096 - len(stored_nmer_weights['fdse']))

    nmer_df = pd.DataFrame(stored_nmer_weights)
    nmer_df = nmer_df[['nmer', 'use', 'cse', 'dse', 'fdse']]

    nmer_df.to_csv(os.path.join(outputPath, fileName + '.csv'), index=False, sep='\t')