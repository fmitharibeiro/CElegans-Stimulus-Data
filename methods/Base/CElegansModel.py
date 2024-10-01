import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, TimeDistributed, Dense, Input

class CElegansModel:
    def __init__(self, seed, input_size=4, num_hidden_layers=16, output_size=4):
        self.seed = seed
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.lr = 1
        self.batch_size = 32
        self.epochs = 1000
        self.kwargs = {}

        # Input layer for the sequence data (X)
        inputs = Input(shape=(None, input_size), name="input_sequence")

        # Input layer for the initial hidden state
        initial_state_input = Input(shape=(self.num_hidden_layers,), name="initial_state_input")

        # GRU layer that accepts an initial state
        gru_layer = GRU(self.num_hidden_layers, return_sequences=True, return_state=True)
        gru_output, gru_hidden = gru_layer(inputs, initial_state=initial_state_input)

        # TimeDistributed Dense layer for predictions
        output_layer = TimeDistributed(Dense(self.output_size))(gru_output)

        # Define the model that takes both the sequence and the initial state as inputs
        self.model = Model(inputs=[inputs, initial_state_input], outputs=[output_layer, gru_hidden])

        self.opt_function = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.model.compile(optimizer=self.opt_function, loss='mean_squared_error')

    def fit(self, X, y):
        tf.random.set_seed(self.seed)
        self.model.fit(
            {'input_sequence': np.array(X), 'initial_state_input': np.zeros((X.shape[0], self.num_hidden_layers))},  # Zero initial state for training
            np.array(y),
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1
        )
    
    def predict(self, X, initial_state=None, return_hidden=False, *args, **kwargs):
        """
        Makes a prediction, with optional initial hidden state.
        Returns predictions and optionally the hidden state.
        """
        X = np.array(X)
        
        # If no initial state is provided, use zeros
        if initial_state is None:
            initial_state = np.zeros((X.shape[0], self.num_hidden_layers))

        # Use the full model to predict, passing both the input and initial state
        predictions, hidden_state = self.model.predict(
            {'input_sequence': X, 'initial_state_input': initial_state},
            batch_size=self.batch_size,
            verbose=kwargs.get('verbose', 0)
        )

        if return_hidden:
            return predictions, hidden_state
        return predictions

    def __call__(self, X, initial_state=None, return_hidden=False, *args, **kwargs):
        """
        If initial_state is provided, return predictions and hidden state.
        Otherwise, return just the predictions and hidden state.
        """
        if initial_state is not None and len(initial_state.shape) == 3:
            initial_state = initial_state[0]
        return self.predict(X, initial_state=initial_state, return_hidden=return_hidden, **kwargs)

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
