import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam


class CElegansModel:
    def __init__(self, seed, num_hidden_layers=8, output_size=4):
        self.seed = seed # Does nothing
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.opt_function = None
        self.lr = 1
        self.batch_size = 32
        self.epochs = 1000
        self.kwargs = {}

        self.model = Sequential()
        self.model.add(GRU(self.num_hidden_layers, return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.output_size)))

        self.param_grid = {
            'lr': ("suggest_loguniform", 1e-5, 5e-2)
            # 'batch_size': ("suggest_categorical", [4, 8, 16, 32])
        }
    
    def fit(self, X, y):
        self.opt_function = Adam(learning_rate=self.lr)

        self.model.compile(optimizer=self.opt_function, loss='mean_squared_error')

        tf.random.set_seed(self.seed)

        self.model.fit(
            np.array(X),
            np.array(y),
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1
        )
    
    def predict(self, X, *args, **kwargs):
        return self.model.predict(np.array(X), batch_size=self.batch_size, verbose=kwargs.get('verbose'))
    
    def __call__(self, X, *args, **kwargs):
        if args:
            return self.predict(X, return_state=True)
        return self.predict(X)
    
    def save(self, filename):
        self.model.save(filename)

    def set_params(self, **params):
        if not params:
            return self
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self

    def get_params(self):
        return {attr: getattr(self, attr)
                for attr in dir(self)
                if not callable(getattr(self, attr)) and not attr.startswith("__")}
