from turtle import title
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import os

def plotLossfunctionPerEpochs(history, fileName):
    plt.figure(figsize=(15, 15))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(fileName)

def plotScatterWithAxis(prediction, y_test, outputPath, fileName, model_version, data_version, plotName):
    r_val, p_val = pearsonr(prediction, y_test)

    plt.figure(figsize=(15, 15))
    
    plt.scatter(prediction, y_test, color='black', s=5, alpha=0.05)
    plt.scatter(prediction, y_test, color='black', s=np.pi * (2 * np.ones(1))**2, alpha=0.05)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    min_x = max(np.min(prediction), np.min(y_test))
    max_x = min(np.max(prediction), np.max(y_test))
    min_y = max(np.min(prediction), np.min(y_test))
    max_y = min(np.max(prediction), np.max(y_test))
    plt.plot([min_x, max_x], [min_y, max_y], alpha=0.5, color='darkblue', linewidth=3)

    plt.axis([np.min(prediction), np.max(prediction), np.min(y_test), np.max(y_test)])

    plt.xlabel('Pred Proximal Usage', fontsize=14)
    plt.ylabel('True Proximal Usage', fontsize=14)
    
    RMSE = sqrt(mean_squared_error(y_test.tolist(), prediction.tolist()))

    title_text = '{plotName} model = {model}, dataVersion = {dataversion}  (R^2 = {r2}, RMSE = {rmse}, n = {n})'
    title = title_text.format(plotName=plotName, 
                                model=model_version, 
                                dataversion=data_version,
                                r2=round(r_val * r_val, 2), 
                                rmse=RMSE, 
                                n=len(y_test))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outputPath, fileName))