import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler

OPEN_COLUMN = 1
HIGHT_COLUMN = 2
LOW_COLUMN = 3
CLOSE_COLUMN = 4
VOLUME_COLUMN = 5

DAYS_TO_TEST = 30
PREVIOUS_DAYS = 60

class DataBuilder:
    def __init__(self, data_set, entry_columns, prediction_column):
        self.data_set = data_set
        self.entry_columns = entry_columns
        self.prediction_column = self.map_column(prediction_column)
        
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        
    def build_data(self):
        self.calculate_first_training_row()
        
        training_data = self.load_training_data()
        training_data_scaled = self.scaler.fit_transform(training_data)
        self.training_entry_set = self.create_entry_set(training_data_scaled)
        
        training_result = self.load_training_result()
        training_result_scaled = self.scaler.fit_transform(training_result)
        self.training_result_set = self.create_training_result_set(training_result_scaled)
        
        test_data = self.load_test_data()
        test_data_scaled = test_data.reshape(-1,training_data_scaled.shape[1])
        test_data_scaled = self.scaler.transform(test_data_scaled)
        self.test_entry_set = self.create_entry_set(test_data_scaled)
        
        self.real_test_results = self.load_real_test_results()
    
    def map_column(self, column):
        if(column == 'open'):
            return OPEN_COLUMN
        if(column == 'close'):
            return CLOSE_COLUMN
        if(column == 'hight'):
            return HIGHT_COLUMN
        if(column == 'low'):
            return LOW_COLUMN
    
    def calculate_first_training_row(self):
        self.training_row_from = 0
        if('rsi' in self.entry_columns):
            self.training_row_from = 14
        if('macd' in self.entry_columns):
            self.training_row_from = 26
    
    def load_training_data(self):
        self.training_row_from
        training_row_to = self.data_set.shape[0] - DAYS_TO_TEST
        
        return self.load_entry_data(self.training_row_from, training_row_to)
    
    def load_test_data(self):
        test_row_from = self.data_set.shape[0] - DAYS_TO_TEST - PREVIOUS_DAYS
        test_row_to = self.data_set.shape[0]
        
        return self.load_entry_data(test_row_from, test_row_to)
    
    def load_entry_data(self, row_from, row_to):
        entry = []
        for column_name in self.entry_columns:
            entry.append(self.load_column(column_name, row_from, row_to))
        
        return np.column_stack(tuple(entry))
          
    def load_column(self, column_name, row_from, row_to):
        transpose_full_close = self.data_set.iloc[:, CLOSE_COLUMN:CLOSE_COLUMN+1]
        full_close = transpose_full_close.transpose().iloc[0,:]
        
        if(column_name == 'macd'):            
            macd = ta.trend.MACD(full_close).macd()
            return macd.to_frame().iloc[row_from:row_to, 0:1]
        
        if(column_name == 'rsi'):
            rsi = ta.momentum.RSIIndicator(full_close).rsi()
            return rsi.to_frame().iloc[row_from:row_to, 0:1]
        
        if(column_name == 'open'):
            return self.data_set.iloc[row_from:row_to, OPEN_COLUMN:OPEN_COLUMN+1]
        
        if(column_name == 'hight'):
            return self.data_set.iloc[row_from:row_to, HIGHT_COLUMN:HIGHT_COLUMN+1]
        
        if(column_name == 'low'):
            return self.data_set.iloc[row_from:row_to, LOW_COLUMN:LOW_COLUMN+1]
        
        if(column_name == 'close'):
            return self.data_set.iloc[row_from:row_to, CLOSE_COLUMN:CLOSE_COLUMN+1]
        
        if(column_name == 'volume'):
            return self.data_set.iloc[row_from:row_to, VOLUME_COLUMN:VOLUME_COLUMN+1]
    
    def create_entry_set(self, scaled_data):
        raw_entry_set = []

        for i in range(PREVIOUS_DAYS, len(scaled_data)):
            raw_entry_set.append(scaled_data[i-PREVIOUS_DAYS:i])
        
        raw_array = np.array(raw_entry_set)
        
        new_shapes = (raw_array.shape[0], raw_array.shape[1], raw_array.shape[2])
        reshaped_set = np.reshape(raw_array, new_shapes)
        
        return reshaped_set
        
    def create_training_result_set(self, raw_array):
        raw_result_set = []
        for i in range(PREVIOUS_DAYS, len(raw_array)):
            raw_result_set.append(raw_array[i, 0])
    
        return np.array(raw_result_set)
    
    def load_training_result(self):
        row_to = self.data_set.shape[0] - DAYS_TO_TEST
    
        return self.data_set.iloc[self.training_row_from : row_to,
                         self.prediction_column : self.prediction_column+1].values
    
    def load_real_test_results(self):
        row_from = self.data_set.shape[0] - DAYS_TO_TEST
        row_to = self.data_set.shape[0]
    
        return self.data_set.iloc[row_from : row_to,
                             self.prediction_column : self.prediction_column+1].values
    
    def inverse_transform(self, scaled_predicted_results):
        return self.scaler.inverse_transform(scaled_predicted_results)