# Trading Recurrent Neural Networks Builder
This project has the main goal of apply the matching learning technic to predict trading data of companies which operates on the stock market, using for this purpose only the past trading data of the company (each company will use only its own past data).

On matching learning, a Recurrent Neural Network (RNN) is a neural network that creates a topology composed by multiple neural networks which can take multiple vectors as input and optionally generates multiple outputs (could generate one per layer). Each layer of this topology take the knowledge obtained on previous layer, and its own inputs and process it as a simple neural network, generating predictions that could be dump to an output vector and throw this knowledge to follow layer of the topology (or both). That means, the recurrent neural network keep a memory of what happen on previous layer and throw this memory by the topology.

This kind of neural network will be the selected one to make predictions of trading values of a company. A company close a trading day with some trading values (open price, close price, volume, etc); this data keep a relationship between them, and the value of one could be influenced by the others. Because of this relationship, the knowledge obtained of process a trading field, should be used by other fields.

Of course, a number of external actors will influence on this trading data too, and on a higher weight, but are out os scope of this project, that only take the past data of same company to make predictions (is a matching learning project, no a trading one).

Each company has each own reality, and the way as the company evolve is different on each one; for this reason can not been defined a neural network that could be use to obtain predictions of any company, given its past trading data. Instead that, a specific neural network should been defined for each company, that will be trained only with the data of the specific company and, probably, the configuration of initial parameters of the neural network changes from one company to other to obtain the more efficient results.

To build and train a RNN model, previously should been configured some parameters, like the neuron activate function, learning rate, batch size, etc. As mentioned, given the same configuration, greats results can be obtained for one company and be disastrous for another; so should been adjusted the better configuration for any company (and probably for different time period).

This project stablish a system that, given a data set composed by trading data of a company for a period, tries different configuration values, and stablish the optimal configuration for the RNN. Then build a RNN sequential model, with the previously calculated configuration values and generates as output the "h5" file of the keras model and a graphic with obtained results. 

The output graphic obtained on this step are expected to be good, because the configuration was adjusted comparing predicted with real result on the same data set that uses to build this graphic. The real challenge is to use the sequential obtained model to make predictions of trading data of the same company on a future period (that does not be present on the original input data set) and compare this predicted values to real ones. If obtained results are acceptable, the keras model built is suitable to predict future trading values of this company.

## Goals
Given previous approach, the goals of the projects are follow ones:
- Given a data set with the trading close data of a company, obtain the more efficient configuration of a recurrent neural network to predict each trading field.

- Build a training recurrent neural network model for each trading field of company given its data set.

- Be able to easily changes the inputs vectors to configure and build a RNN of a trading field of a company.

- Be able to test the RNN models with a data set of the company different than uses to build the model.

- Stablish a system to easily configure and build multiple models for different companies, given its entry data sets.

## Get Started
Follow this steps to get started with the solution, and create a recurrent neural network that predicts the trading data of a company.

### Input data
On the Tools folder, you can find a script (wget-raw-data) to load trading data of the Spanish trading market from the web site: www.megabolsa.com
Edit it to stablish a period and a output folder and execute in order to load data to test the solution. You can add a filter in order to obtain data only of few companies which ones you want to test the solution (not need at the beginning all Spanish market trading data).

Obtained data are on a raw format, not with the needed structure to use as input for the RNN builder. You can use the "build_trading_datasets.py" program from the Tools folder to transform the raw data on csv files with the input data structured as the RNN builder expects. Edit it first in order to customize the input and output folders.

At this point you have at output configured folder some csv files which can be use to build RNN models of selected companies to predict trading data.

Review the generated files because, depends on the company, its possible that the data are not complete, or be little data to build a a RNN successfully. To obtains acceptable results, you need a data set with at least 1000/1200 rows (days). To test the solution, just choose a company which data are complete and follow up.

## Building RNN models
Given the entry csv data set of a company, put this file on an empty folder (generated one is ok); you can use multiple files, from multiples companies, but it will take more time. 

Open the trading_rnns_builder.py file and customize (if needed) the input and output folders. On the same execution, the program will build RNNs of four trading fields of the company (open, close, high and low). For each one, you can edit the input vectors that will use to build the RNN model. Probably this inputs could be change for different companies, to obtain the more accurate results; you can try with different input values and compare the results.

Additionally, you can uses as input vectors trading indicators like MACD. Currently only can choose the MACD and the RSI indicators, but thanks to Technical Analysis Library in Python (https://github.com/bukosabino/ta) you can easily add more indicators if you need, just improving the data_builder.py class.

Once you had established the input vectors, just running the trading_rnns_builder program, it will generates as output RNNs model of the different trading fields (as h5 file) and a png with a graphic of obtained predictions of part of data set compared with real results.

## Test the RNN models
As mentioned, the obtained graphic on previous step is not a good indicator about the model accuracy, because the RNN configuration was adjusted with same data that are represented on the graphic. The model should be testing with different data that used to build it. 

For that purpose, you can use the "conpany_rnns_tester.py" program from Tools folder to make prediction of data of the same company that not included on the csv used to build the RNN model. For that, you can, for example, use a subset of the original csv to build the RNN model and use full one to test it.