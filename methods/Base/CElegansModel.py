import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, TimeDistributed, Dense, Input

class CElegansModel:
    def __init__(self, seed, input_size=4, num_hidden_layers=16, output_size=4):
        self.seed = seed
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.opt_function = None
        self.lr = 1
        self.batch_size = 32
        self.epochs = 1000
        self.kwargs = {}

        # Input layer
        inputs = Input(shape=(None, input_size))

        # GRU layer (defined as a layer, not called here)
        self.gru_layer = GRU(self.num_hidden_layers, return_sequences=True, return_state=True)

        # Call the GRU layer on inputs and get both output and hidden state
        gru_output, gru_hidden = self.gru_layer(inputs)

        # TimeDistributed Dense layer (defined as a layer object, not computed yet)
        self.output_layer = TimeDistributed(Dense(self.output_size))

        # Call the output layer with the GRU output to get the final output
        outputs = self.output_layer(gru_output)

        # Define the model with input and output (only output sequence is used for the final output)
        self.model = Model(inputs=inputs, outputs=[outputs, gru_hidden])

        self.param_grid = {
            'lr': ("suggest_loguniform", 1e-5, 5e-2)
        }
    
    def fit(self, X, y):
        self.opt_function = tf.keras.optimizers.Adam(learning_rate=self.lr)
        # Compile the model to optimize for the first output (prediction)
        self.model.compile(optimizer=self.opt_function, loss='mean_squared_error')

        tf.random.set_seed(self.seed)

        self.model.fit(
            np.array(X),
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
        
        if initial_state is not None:
            # Manually call the GRU layer with initial_state and extract the hidden state and output
            gru_output, hidden_state = self.gru_layer(X, initial_state=initial_state)

            # Pass only the output to the TimeDistributed layer to get predictions
            predictions = self.output_layer(gru_output)

            return predictions, hidden_state
        else:
            # Standard prediction (returns both predictions and hidden state)
            predictions, hidden_state = self.model.predict(X, batch_size=self.batch_size, verbose=kwargs.get('verbose'))
            if return_hidden:
                return predictions, hidden_state
            return predictions

    def __call__(self, X, initial_state=None, return_hidden=False, *args, **kwargs):
        """
        If initial_state is provided, return predictions and hidden state.
        Otherwise, return just the predictions and hidden state.
        """
        if len(initial_state.shape) == 3:
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
