from datasets import CEHandler


datasets = {"CE": CEHandler()}

def fetch_data(dataset):
    """
    Loads the dataset, performing a 80-20 train-test split
    """
    if not dataset in datasets:
        raise AssertionError

    handler = datasets[dataset]
    X_train, y_train, X_test, y_test = handler.fetch_data()
    return {'train':(X_train, y_train), 
     'test':(X_test, y_test)}