from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import os
import numpy as np

from aparent_data_plasmid import load_data
from aparent_model_plasmid_large_lessdropout import load_aparent_model
from common.PlotHelper import plotLossfunctionPerEpochs, plotScatterWithAxis
from neuralNetwork.KerasModelHelper import trainModel, saveModel, loadModel
from ArgumentsParser import getParsedArguments

basePath, outputPath, dataPath, sampleSize = getParsedArguments()

#Trainer parameters
load_saved_model = False
save_dir_path = os.path.join(outputPath, 'saved_models')
load_name_suffix = 'all_libs_no_sampleweights'
save_name_suffix = 'all_libs_no_sampleweights'
epochs = 5 #15
batch_size = 20
use_sample_weights = False

valid_set_size = 0.05
test_set_size = 0.05

kept_libraries = None

data = load_data(batch_size=batch_size, file_path=dataPath, valid_set_size=valid_set_size, test_set_size=test_set_size, data_version='', kept_libraries=kept_libraries, sampleSize=sampleSize)
models = load_aparent_model(batch_size, use_sample_weights=use_sample_weights)
plasmid_model_prefix, plasmid_model = models[0]
loss_model_prefix, loss_model = models[1]
trainingData = data[0]
predictionData = data[1]
# predictionData = data

checkpoint_dir = os.path.join(outputPath, 'model_checkpoints')
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

opt = tf.keras.optimizers.SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)

loss_model.compile(loss=lambda true, pred: pred, optimizer=opt)

callbacks = [
    ModelCheckpoint(os.path.join(checkpoint_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq="epoch"),
    EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=3, verbose=0, mode='auto')
]

loss_model, loss_hist  = trainModel(trainingData, loss_model, epochs, callbacks, True, 20)

saveModel(save_dir_path, loss_model, loss_model_prefix, save_name_suffix)

plotLossfunctionPerEpochs(loss_hist, os.path.join(outputPath, 'loss_hist.png'))

y_test_pred = loss_model.predict(trainingData['test'])

# prediction= np.empty(0)
# for pred in y_test_pred:
#     prediction = np.append(trainingData, pred[0])

# y_test= np.empty(0)
# for list_item in trainingData['test']:
#     for item in list_item[1][0]:
#         y_test = np.append(y_test, item[0])

# plotScatterWithAxis(prediction, y_test, outputPath, 'lossPrediction.png') 

opt = tf.keras.optimizers.SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)

plasmid_model.compile(loss=lambda true, pred: pred, optimizer=opt)

callbacks = [
    ModelCheckpoint(os.path.join(checkpoint_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq="epoch"),
    EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=3, verbose=0, mode='auto')
]

plasmid_model, plasmid_hist = trainModel(predictionData, plasmid_model, epochs, callbacks, True, 20)
saveModel(save_dir_path, plasmid_model, plasmid_model_prefix, save_name_suffix)
plotLossfunctionPerEpochs(plasmid_hist, os.path.join(outputPath, 'plasmid_hist.png'))

# plasmid_model = loadModel('../output/aparent_plasmid_iso_cut_distalpas_large_lessdropout_all_libs_no_sampleweights.h5')
y_test_pred = plasmid_model.predict(predictionData['test'])

prediction= np.empty(0)
for pred in y_test_pred[0]:
    prediction = np.append(prediction, pred[0])

y_test= np.empty(0)
for list_item in predictionData['test']:
    for item in list_item[1][0]:
        y_test = np.append(y_test, item[0])

plotScatterWithAxis(prediction, y_test, outputPath, 'plasmidPrediction.png') 