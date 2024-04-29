from .TimeSHAP_Expl import TimeSHAP_Explainer


explainer_names = {
    'TimeSHAP':TimeSHAP_Explainer,
}

def fetch_explainer(name, **kwargs):
    if name in explainer_names:
        return explainer_names[name](**kwargs)
    else:
        print(f"Explainer {name} not found. (Base model running?)")
        return None
