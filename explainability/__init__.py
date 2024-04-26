from .TimeSHAP_Expl import TimeSHAP_Explainer


method_names = {
    'TimeSHAP':TimeSHAP_Explainer,
}

def fetch_explainer(name, **kwargs):
    if name in method_names:
        return method_names[name](**kwargs)
    else:
        print(f"Method {name} not found. (Base model running?)")
        return None
