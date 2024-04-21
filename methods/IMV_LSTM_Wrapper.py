from .IMV_LSTM.networks import IMVTensorLSTM, IMVFullLSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class IMV_LSTM_Wrapper:
    def __init__(self, num_hidden_layers=3, init_std=0.02, batch_size=32, epochs=10, seed=42):
        self.model = None
        self.seed = seed
        self.kwargs = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_hidden_layers = num_hidden_layers
        self.optimizer_name = "Adam"
        self.lr = 1
        self.init_std = init_std
        self.batch_size = batch_size
        self.epochs = epochs

        self.param_grid = {
            'num_hidden_layers': ("suggest_int", 100, 200),  # Number of hidden layers
            # 'dropout_prob': ("suggest_uniform", 0.1, 0.2)  # Dropout probability
            'optimizer_name': ("suggest_categorical", ["Adam", "RMSprop", "SGD"]),
            'lr': ("suggest_float", 1e-5, 1e-1),
            'init_std': ("suggest_float", 1e-2, 1),
            'batch_size': ("suggest_int", 1, 32),
            'epochs': ("suggest_int", 2, 10),
        }
    
    def fit(self, X, y):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        input_dim = X_tensor.shape[2]
        output_dim = 1
        self.model = IMVTensorLSTM(input_dim, output_dim, self.num_hidden_layers, self.init_std).to(self.device)
        criterion = nn.MSELoss()
        optimizer = getattr(optim, self.optimizer_name)(self.model.parameters(), lr=self.lr)
        # optimizer = torch.optim.Adam(self.model.parameters()) # learning rate?

        progress_bar = tqdm(total=len(train_dataloader) * self.epochs, desc="Training", leave=False)
        for epoch in range(self.epochs):
            self.model.train()
            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                outputs, _, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item(), epoch=epoch + 1)
                progress_bar.update(1)  # Update the progress bar
        progress_bar.close()
    
    def predict(self, X):
        # Convert data to PyTorch tensor and move to device
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Predict
        with torch.no_grad():
            self.model.eval()
            predictions_tuple = self.model(X_tensor)

        # Separate the values in the tuple
        prediction_main = predictions_tuple[0].cpu().numpy()
        alpha_prediction = predictions_tuple[1].cpu().numpy()
        beta_prediction = predictions_tuple[2].cpu().numpy()

        # Store other predictions in self.kwargs
        self.kwargs['alpha'] = alpha_prediction
        self.kwargs['beta'] = beta_prediction

        return prediction_main
    
    def set_params(self, **params):
        if not params:
            return self
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self




class NeuralNetwork:
    def __init__(self, num_hidden_layers=1, dropout_prob=0.5):
        self.num_hidden_layers = num_hidden_layers
        self.dropout_prob = dropout_prob
        self.model = None
        self.kwargs = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y, batch_size=32, epochs=10):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)  # Assuming y is 1D array

        X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

        train_dataset = TensorDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        input_dim = X_tensor.shape[1]
        self.model = SimpleNN(input_dim, self.num_hidden_layers, self.dropout_prob).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters())

        for epoch in range(epochs):
            self.model.train()
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for batch_X, batch_y in progress_bar:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())

    def predict_proba(self, X):
        # Convert data to PyTorch tensor and move to device
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Predict logits
        with torch.no_grad():
            self.model.eval()
            logits = self.model(X_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        return np.hstack((1 - probabilities, probabilities))