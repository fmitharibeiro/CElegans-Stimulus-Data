from .TimeSHAP_Wrapper import TimeSHAP_Explainer
from .SeqSHAP_Wrapper import SeqSHAP_Explainer


explainer_names = {
    'TimeSHAP':TimeSHAP_Explainer,
    'SeqSHAP':SeqSHAP_Explainer,
}

def fetch_explainer(name, **kwargs):
    if name in explainer_names:
        return explainer_names[name](**kwargs)
    else:
        print(f"Explainer {name} not found. (Base model running?)")
        return None
