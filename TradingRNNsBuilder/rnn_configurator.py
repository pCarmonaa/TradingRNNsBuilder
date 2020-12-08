from rnn_wrapper import RNNWrapper
from dtos.rnn_configuration import RNNConfiguration
from dtos.trading_data import TradingData
import numpy as np

import copy

class RNNConfigurator:
    def __init__(self, trading_data):
        self.trading_data = trading_data

    def configure(self):
        configuration = RNNConfiguration()
        configuration.epochs = 5
        configuration = self.stablish_base_configuration(configuration)
        configuration = self.stablish_bach(configuration)
        configuration = self.stablish_beta_1(configuration)
        configuration = self.stablish_beta_2(configuration)
        configuration = self.stablish_patience(configuration)
        
        return configuration

    def stablish_base_configuration(self, configuration):
        functions = ['sigmoid', 'relu']
        learning_rates = [0.1, 0.01, 0.001]

        aux_configuration = copy.copy(configuration)
        for function in functions:
            for learning_rate in learning_rates:
                aux_configuration.activation_function = function
                aux_configuration.learning_rate = learning_rate
                rnn = self.run_simulation(aux_configuration)
                
                aux_configuration.tend_score = self.evaluate_tend_results(rnn)
                if(aux_configuration.tend_score < configuration.tend_score):
                    configuration = copy.copy(aux_configuration)

        return configuration

    def stablish_bach(self, configuration):
        aux_configuration = copy.copy(configuration)
        
        batch_sizes = [16,32,64,128]
        for batch_size in batch_sizes:
            aux_configuration.batch_size = batch_size
            rnn = self.run_simulation(aux_configuration)
            
            aux_configuration.tend_score = self.evaluate_tend_results(rnn)
            if(aux_configuration.tend_score < configuration.tend_score):
                configuration = copy.copy(aux_configuration)

        return configuration

    def stablish_beta_1(self, configuration):
        aux_configuration = copy.copy(configuration)

        beta_ones = [0.5, 0.8, 0.9]
        for beta_1 in beta_ones:
            aux_configuration.beta_1 = beta_1
            rnn = self.run_simulation(aux_configuration)

            aux_configuration.tend_score = self.evaluate_tend_results(rnn)
            if(aux_configuration.tend_score < configuration.tend_score):
                configuration = copy.copy(aux_configuration)

        return configuration

    def stablish_beta_2(self, configuration):
        aux_configuration = copy.copy(configuration)

        beta_twos = [0.9, 0.99]
        for beta_2 in beta_twos:
            aux_configuration.beta_2 = beta_2
            rnn = self.run_simulation(aux_configuration)

            aux_configuration.tend_score = self.evaluate_tend_results(rnn)
            if(aux_configuration.tend_score < configuration.tend_score):
                configuration = copy.copy(aux_configuration)

        return configuration

    def stablish_patience(self, configuration):
        aux_configuration = copy.copy(configuration)

        patiences = [10, 50]
        for patience in patiences:
            aux_configuration.patience = patience
            rnn = self.run_simulation(aux_configuration)

            aux_configuration.tend_score = self.evaluate_tend_results(rnn)
            if(aux_configuration.tend_score < configuration.tend_score):
                configuration = copy.copy(aux_configuration)

        return configuration

    def run_simulation(self, configuration):
        _trading_data = copy.copy(self.trading_data)
        _trading_data.config = configuration
        rnn = RNNWrapper(_trading_data)
        rnn.build()
        rnn.fit()
        rnn.predict()
        rnn.load_real_results()

        return rnn
    
    def evaluate_tend_results(self, rnn):
        diffs = []
        for i in range(rnn.real_results.shape[0]):
            dif = abs(rnn.predicted_results[i][0] - rnn.real_results[i][0])
            diffs.append(dif)

        return np.average(diffs)    
