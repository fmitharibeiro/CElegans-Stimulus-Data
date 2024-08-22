from .IMV_LSTM_Wrapper import IMV_LSTM_Wrapper
from .Base.CElegansModel import CElegansModel
from .Base.CElegansModel_Torch import CElegansModel_Torch

method_names = {
    'IMV-LSTM':IMV_LSTM_Wrapper,
    'BaseCE':CElegansModel,
    'BaseCE_Torch':CElegansModel_Torch
}

def fetch_method(name, seed, **kwargs):
    if name in method_names:
        input_size = getattr(kwargs.get('other_args'), 'input_size')
        num_hidden_layers = getattr(kwargs.get('other_args'), 'num_hidden_layers')
        output_size = getattr(kwargs.get('other_args'), 'output_size')
        return method_names[name](seed=seed, input_size=input_size, num_hidden_layers=num_hidden_layers, output_size=output_size)
    else:
        print(f"Method {name} not found. (Post-hoc methods running?)")
        return None