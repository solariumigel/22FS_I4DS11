import os
from tensorflow import keras
from common.PositionShifter import PositionShifter
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import tensorflow as tf
from common.PlotHelper import plotLossfunctionPerEpochs, plotScatterWithAxis

def trainModel(data, model, epochs, callbacks, use_multiprocessing=False, workers=1):
    history = model.fit(data['train'],
                    validation_data=data['valid'],
                    epochs=epochs,
                    # use_multiprocessing=use_multiprocessing,
                    # workers=workers,
                    callbacks=callbacks)
    return model, history

def saveModel(save_dir, model, fileName):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_name = fileName + '.h5'
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

def loadModel(modelPath):
    print(modelPath)
    model = keras.models.load_model(modelPath)
    return model

def LoadAndTrain(data, output_Path, model, model_prefix, model_version, data_version, epochs=5):
    save_dir_path = os.path.join(output_Path, 'saved_models')
    
    checkpoint_dir = os.path.join(output_Path, 'model_checkpoints')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    opt = tf.keras.optimizers.SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt)

    callbacks = [
        ModelCheckpoint(os.path.join(checkpoint_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq="epoch"),
        EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=3, verbose=0, mode='auto'),
        TensorBoard(log_dir=os.path.join(output_Path, 'Log'), histogram_freq=1),
    ]

    model, plasmid_hist = trainModel(data, model, epochs, callbacks, True, 20)

    fileName = 'aparent_' + model_prefix + '_' + model_version + '_' + data_version
    saveModel(save_dir_path, model, fileName)
    plotLossfunctionPerEpochs(plasmid_hist, os.path.join(output_Path, fileName + '.png'))
    return model

def predict(model, datagenerator, outputPath, model_version, data_version):
    y_test_pred = model.predict(datagenerator['test'])

    prediction= np.empty(0)
    for pred in y_test_pred:
        prediction = np.append(prediction, pred[0])
    
    y_test= np.empty(0)
    for list_item in datagenerator['test']:
        for item in list_item[1][0]:
            y_test = np.append(y_test, item[0])

    fileName = model_version + "_" + data_version +'Prediction.png'
    plotScatterWithAxis(prediction, y_test, outputPath, fileName, model_version, data_version, model_version)

def get_bellcurve_shifter() :
    shift_range = (np.arange(71, dtype=np.int) - 35)
    shift_probs = np.zeros(shift_range.shape[0])
    shift_probs[:] = 0.1 / float(shift_range.shape[0] - 11)
    shift_probs[int(shift_range.shape[0]/2)] = 0.5
    shift_probs[int(shift_range.shape[0]/2)-5:int(shift_range.shape[0]/2)] = 0.2 / 5.
    shift_probs[int(shift_range.shape[0]/2)+1:int(shift_range.shape[0]/2)+1+5] = 0.2 / 5.
    shift_probs /= np.sum(shift_probs)
    
    return PositionShifter(shift_range, shift_probs)

def iso_normalizer(t) :
    iso = 0.0
    if np.sum(t) > 0.0 :
        iso = np.sum(t[80: 80+25]) / np.sum(t)
    
    return iso

def cut_normalizer(t) :
    cuts = np.concatenate([np.zeros(205), np.array([1.0])])
    if np.sum(t) > 0.0 :
        cuts = t / np.sum(t)
    
    return cuts