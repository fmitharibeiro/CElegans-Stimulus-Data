from .IMV_LSTM_Wrapper import IMV_LSTM_Wrapper
from .Base.CElegansModel import CElegansModel

method_names = {
    'IMV-LSTM':IMV_LSTM_Wrapper,
    'BaseCE':CElegansModel
}

def fetch_method(name, seed):
    if name in method_names:
        return method_names[name](seed=seed)
    else:
        print(f"Method {name} not found. (Post-hoc methods running?)")
        return None