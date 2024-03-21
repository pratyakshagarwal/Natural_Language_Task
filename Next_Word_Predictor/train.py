import numpy as np
import pandas as pd
import tensorflow as tf
from data import inputs, categorial_targets
from model import GRU_Model
from Utils.callbacks_utils import CustomTensorflowLogs, CustomEarlyStopping, CustomCSVLogger, CustomLearningRateScheduler
from Utils.evaluation_utils import plot_comfusion_matrix, plot_roc_curve, plot_model_accuracy, plot_model_loss
from keras.metrics import Accuracy, AUC

if __name__ == '__main__':
    # create a instnace of Rnn_model
    model = GRU_Model(metrics=[Accuracy(name='acc'), AUC(name='auc')])
    
    # trying to print summary of model
    print(model.get_summary())

    # parameters
    log_dir = './Next_Word_Predictor/logs/'
    epochs = 100

    # created callbacks
    tensorflow_callback = CustomTensorflowLogs(log_dir=log_dir).make_callback()
    earlystopping_callback = CustomEarlyStopping().make_callback()

    # Create the CSVLogger callback using its make_callback method
    csv_callback = CustomCSVLogger().make_callback(filename='logs.csv')
    callbacks = [tensorflow_callback, earlystopping_callback, csv_callback]

    # train the model 
    history = model.train(train_data=inputs, train_label=categorial_targets, epochs=epochs, callbacks=callbacks)
    
    # saving the model
    model.save_model(filename='Next_Word_Predictor/next_word_prediction_model')
    
    # plotting the graphs for more details
    plot_model_loss(history=history)
    plot_model_accuracy(history=history)
