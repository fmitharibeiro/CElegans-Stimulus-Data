from .IMV_LSTM import IMVTensorLSTM, IMVFullLSTM


class IMV_LSTM_Wrapper:
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        self.model = IMVTensorLSTM(input_dim, output_dim, n_units, init_std)
    
    def fit(self, x):
        pass

