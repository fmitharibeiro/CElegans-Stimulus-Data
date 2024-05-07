



class SeqSHAP_Explainer:
    def __init__(self, model=None, dataset:str=""):
        self.dataset = dataset
        self.model = model
        self.index = 0
        self.save_dir = f"plots/{self.dataset}/SeqSHAP"
        self.f = lambda x: self.model.predict(x)[:, :, self.index] # TODO: Probably expand dims on left

    def __call__(self, X, *args, **kwds):
        pass