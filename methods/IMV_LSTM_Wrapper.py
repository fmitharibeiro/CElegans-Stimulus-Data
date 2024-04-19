from .IMV_LSTM import IMVTensorLSTM, IMVFullLSTM
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class IMV_LSTM_Wrapper:
    def __init__(self, num_hidden_layers, init_std=0.02):
        self.model = None
        self.num_hidden_layers = num_hidden_layers
        self.init_std = init_std
        self.kwargs = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def fit(self, X, y, batch_size=32, epochs=10):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # To check
        X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

        train_dataset = TensorDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        input_dim = X_tensor.shape[1]
        output_dim = y_tensor.shape[1]
        self.model = IMVTensorLSTM(input_dim, output_dim, self.num_hidden_layers, self.init_std).to(self.device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters()) # learning rate?

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