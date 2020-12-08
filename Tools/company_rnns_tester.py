import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.insert(1, './TradingRNNsBuilder')
from data_builder import DataBuilder

import os
import pathlib
from tensorflow.keras.models import load_model

MODEL_FILE = './<set-path-to-h5-file>'
OUTPUT_DATA = './RNNTests/'
DATA_SET = './<set-path-to-csv-dataset>'
ENTRY_COLUMNS = ['open', 'close']

def predict(rnn, data_builder):
    data_builder.build_data()
    entry_set = data_builder.test_entry_set
    scaler_prediction = rnn.predict(entry_set)
    
    return data_builder.inverse_transform(scaler_prediction)

def to_image(predicted_results, real_results, operation, company):
    plt.plot(real_results, color = 'red', label = 'Real Results')
    plt.plot(predicted_results, color = 'blue', label = 'Predicted Results')
    plt.title(company + ' Test Prediction')
    plt.xlabel('Time')
    plt.ylabel(company + ' Price')
    plt.legend()
    plt.savefig(OUTPUT_DATA + '/' + operation + '.png')

operation = MODEL_FILE.split('/')[len(MODEL_FILE.split('/'))-1].split('.')[0]
company = DATA_SET.split('/')[len(DATA_SET.split('/'))-1].split('.')[0]
if(not os.path.isdir(OUTPUT_DATA)):
    pathlib.Path(OUTPUT_DATA).mkdir(parents=True, exist_ok=True)

rnn = tf.keras.models.load_model(MODEL_FILE)
csv_dataset = pd.read_csv(DATA_SET)
data_builder = DataBuilder(csv_dataset, ENTRY_COLUMNS, operation)
predicted_results = predict(rnn, data_builder)
to_image(predicted_results, data_builder.real_test_results, operation, company)