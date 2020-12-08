def auto_str(cls):
    def __str__(self):
        return '%s:\n%s' % (
            type(self).__name__,
            ''.join('\t%s=%s\n' % item for item in vars(self).items())
        )
    cls.__str__ = __str__
    return cls

@auto_str
class RNNConfiguration:
    def __init__(self):
        self.tend_score = 9999
        self.adjustment_score = 9999

        self.layer_units = 60
        self.activation_function = 'sigmoid'
        
        self.learning_rate = 0.01
        self.beta_1 = 0.9
        self.beta_2 = 0.99
        
        self.loss = 'mean_squared_error'

        self.epochs = 100
        self.batch_size = 32

        self.monitor = 'loss'
        self.factor = 0.9
        self.patiente = 50

        self.device = 'cpu'