import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

from data_builder import DataBuilder

class RNNWrapper:
    def __init__(self, trading_data):
        self.config = trading_data.config

        self.company = trading_data.company

        data_set = pd.read_csv(trading_data.csv_dataset)

        self.data_builder = DataBuilder(data_set, trading_data.entry_columns, 
                                        trading_data.prediction_column)
        self.data_builder.build_data()

    def build(self):
       training_entry_set = self.data_builder.training_entry_set
       input_shape = (training_entry_set.shape[1], training_entry_set.shape[2])

       self.sequential = Sequential()
       for i in range(input_shape[1]):
           last_iteration = i == input_shape[1]-1

       self.sequential.add(LSTM(units = self.config.layer_units,
                            recurrent_activation = self.config.activation_function,
                            return_sequences = not last_iteration,
                            input_shape = input_shape))
       self.sequential.add(Dense(units = 1))

       optimizer = Adam(learning_rate=self.config.learning_rate,
                        beta_1=self.config.beta_1,
                        beta_2=self.config.beta_2)
       self.sequential.compile(optimizer = optimizer, loss = self.config.loss)

    def fit(self):
        entry_set = self.data_builder.training_entry_set
        result_set = self.data_builder.training_result_set

        rlrop = ReduceLROnPlateau(monitor=self.config.monitor, 
                                  factor=self.config.factor, 
                                  patience=self.config.patiente)

        with tf.device(self.config.device):
            self.sequential.fit(entry_set,
                                result_set,
                                epochs = self.config.epochs,
                                batch_size = self.config.batch_size,
                                callbacks=[rlrop])
    def predict(self):
        entry_set = self.data_builder.test_entry_set
        scaler_prediction = self.sequential.predict(entry_set)
        self.predicted_results = self.data_builder.inverse_transform(scaler_prediction)
    
    def load_real_results(self): 
        self.real_results = self.data_builder.real_test_results

    def to_image(self):
        plt.clf()
        plt.plot(self.real_results, color = 'red', label = 'Real Results')
        plt.plot(self.predicted_results, color = 'blue', label = 'Predicted Results')
        plt.title(self.company + ' Prediction')
        plt.xlabel('Time')
        plt.ylabel(self.company + ' Price')
        plt.legend()

        return plt

    def save(self, path):
        self.sequential.save(path)