from rnn_configurator import RNNConfigurator
from dtos.trading_data import TradingData
from rnn_wrapper import RNNWrapper
import os
import pathlib
from datetime import datetime

INPUT_DATA = './InputData'
OUTPUT_DATA = './OutputData'

def build_open_preditions(company_file):
    company_name = company_file.split('.')[0]
    trading_data = TradingData(company_name, 
                                INPUT_DATA + '/' + company_file,
                                ['open', 'close'], 'open', None)

    return build_predictions(trading_data, company_name, 'open')

def build_close_predictions(company_file):
    company_name = company_file.split('.')[0]
    trading_data = TradingData(company_name, 
                                INPUT_DATA + '/' + company_file,
                                ['open', 'close'], 'close', None)

    return build_predictions(trading_data, company_name, 'close')
    
def build_high_predictions(company_file):
    company_name = company_file.split('.')[0]
    trading_data = TradingData(company_name, 
                                INPUT_DATA + '/' + company_file,
                                ['high', 'low'], 'high', None)

    return build_predictions(trading_data, company_name, 'high')

def build_low_predictions(company_file):
    company_name = company_file.split('.')[0]
    trading_data = TradingData(company_name, 
                                INPUT_DATA + '/' + company_file,
                                ['high', 'low'], 'low', None)

    return build_predictions(trading_data, company_name, 'low')
    
def build_predictions(trading_data, company_name, operation):
    rnn_configurator = RNNConfigurator(trading_data)
    config = rnn_configurator.configure()
    config.epochs = 100
    trading_data.config = config
    
    rnn = RNNWrapper(trading_data)
    rnn.build()
    rnn.fit()
    rnn.predict()
    rnn.load_real_results()
    plt = rnn.to_image()

    folder = OUTPUT_DATA+'/' + company_name + '-' + datetime.now().strftime('%Y-%m-%d')
    rnn.save(folder + '/' + operation + '.h5')
    plt.savefig(folder + '/' + operation + '.png')
    return rnn.predicted_results

company_files = os.listdir(INPUT_DATA)
for company_file in company_files:
    company_name = company_file.split('.')[0]
    folder = OUTPUT_DATA+'/' + company_name + '-' + datetime.now().strftime('%Y-%m-%d')
    if(not os.path.isdir(folder)):
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    predicted_open = build_open_preditions(company_file)
    predicted_close = build_close_predictions(company_file)
    predicted_high = build_high_predictions(company_file)
    predicted_low = build_low_predictions(company_file)