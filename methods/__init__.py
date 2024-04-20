from .IMV_LSTM_Wrapper import IMV_LSTM_Wrapper

method_names = {
    'IMV-LSTM':IMV_LSTM_Wrapper
}

def fetch_method(name, seed):
    assert name in method_names
    return method_names[name](seed=seed)