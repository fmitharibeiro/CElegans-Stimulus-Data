from .IMV_LSTM_Wrapper import IMV_LSTM_Wrapper
from .TimeShap_Wrapper import TimeShap_Wrapper

from .Base.CElegansModel import CElegansModel

method_names = {
    'IMV-LSTM':IMV_LSTM_Wrapper,
    'TimeSHAP':TimeShap_Wrapper,
    'BaseCE':CElegansModel
}

def fetch_method(name, seed):
    assert name in method_names
    return method_names[name](seed=seed)