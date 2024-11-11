import torch
import torch.nn as nn
import torch.optim as optim


class CElegansModel_Torch(nn.Module):
    def __init__(self, seed, input_size=4, num_hidden_layers=8, output_size=4):
        super(CElegansModel_Torch, self).__init__()
        torch.manual_seed(seed)
        self.seed = seed
        self.num_hidden_layers = num_hidden_layers
        self.output_size = output_size
        self.lr = 1
        self.batch_size = 32
        self.epochs = 1000
        self.kwargs = {}

        # Define the layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=num_hidden_layers, batch_first=True)
        self.fc = nn.Linear(num_hidden_layers, output_size)

        self.opt_function = None

        self.param_grid = {
            'lr': ("suggest_loguniform", 1e-5, 5e-2)
        }

    def forward(self, x, hidden_states=None):
        # Pass through GRU layer with or without hidden states
        if hidden_states is None:
            gru_out, hidden = self.gru(x)
        else:
            gru_out, hidden = self.gru(x, hidden_states)
        
        # Apply the dense layer in a time-distributed manner
        out = self.fc(gru_out)
        
        if hidden_states is not None:
            return out, hidden
        return out

    def fit(self, X, y):
        self.opt_function = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Training loop
        self.train()
        for epoch in range(self.epochs):
            self.opt_function.zero_grad()
            output = self.forward(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            self.opt_function.step()

            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item():.4f}')

    def predict(self, X, hidden_states=None, *args, **kwargs):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)

            if hidden_states is None:
                output = self.forward(X_tensor, hidden_states=hidden_states)
                return output.numpy()
            
            output, hidden = self.forward(X_tensor, hidden_states=hidden_states)
            return output.numpy(), hidden

    def __call__(self, X, hidden_states=None, *args, **kwargs):
        return self.predict(X, hidden_states=hidden_states)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def set_params(self, **params):
        if not params:
            return self
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self

    def get_params(self):
        return {attr: getattr(self, attr)
                for attr in dir(self)
                if not callable(getattr(self, attr)) and not attr.startswith("__")}
