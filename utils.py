from datasets import CEHandler


datasets = {"CE": CEHandler()}

def fetch_data(dataset, reduct):
    """
    Loads the dataset, performing a 80-20 train-test split
    """
    if not dataset in datasets:
        raise AssertionError
    assert 0 <= reduct <= 1, "Reduction factor must be between 0 and 1"

    handler = datasets[dataset]
    X_train, y_train, X_test, y_test = handler.fetch_data()

    red_ind_train = int(X_train.shape[0] * reduct)
    red_ind_test = int(X_test.shape[0] * reduct)
    X_train = X_train[:red_ind_train]
    y_train = y_train[:red_ind_train]
    X_test = X_test[:red_ind_test]
    y_test = y_test[:red_ind_test]

    print(f"Dataset has {red_ind_train} training and {red_ind_test} test samples.")

    return {'train':(X_train, y_train), 'test':(X_test, y_test)}