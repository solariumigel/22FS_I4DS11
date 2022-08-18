from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import os
import numpy as np
import pandas as pd

from ArgumentsParser import getParsedArguments
from common.PlotHelper import plotLossfunctionPerEpochs, plotScatterWithAxis
from neuralNetwork.KerasModelHelper import *
from neuralNetwork.aparent_data_plasmid_datenset import *
from neuralNetwork.aparent_model_plasmid_large_lessdropout import *


basePath = "../service"
dataPath = "../data/"
sampleSize = None
outputPath = "../output/own/NeuralNetwork/20Epoch"
model_version = 'aparent'
data_version = 'Datenset1'

basePath, outputPath, dataPath, sampleSize, filename, dataVersion, modelVersion = getParsedArguments()

#Trainer parameters
epochs = 20
batch_size = 21
valid_set_size = 0.05
test_set_size = 0.05


def getDirection(row):
    return row.split(':')[2]


sampleSize = None
seq_length = 400
seq_startpos = 0

filePath = os.path.join(dataPath, filename)
data = pd.read_csv(filePath, sep=",")

datagenerator = load_data_prediction(data=data, seq_length=seq_length, seq_startpos=seq_startpos, batch_size=batch_size, valid_set_size=valid_set_size, test_set_size=test_set_size, sampleSize=sampleSize)

model_prefix, model = load_aparent_model_Plasmid_withoutLibrary(seq_length=seq_length)
model = LoadAndTrain(data=datagenerator, model_prefix=model_prefix, model=model, output_Path=outputPath, model_version=modelVersion, data_version=dataVersion, epochs=epochs)
predict(model, datagenerator, outputPath, modelVersion, dataVersion)