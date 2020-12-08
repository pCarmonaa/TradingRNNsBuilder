class TradingData:
    def __init__(self, company, csv_dataset, entry_columns, 
                prediction_column, config):
        self.company = company
        self.csv_dataset = csv_dataset
        self.entry_columns = entry_columns
        self.prediction_column = prediction_column
        self.config = config