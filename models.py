import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, dropout_prob=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout_prob),
            
            nn.Conv1d(hidden_size, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout_prob)
        )
        
        # Calculate output size after convolutions
        self.feature_size = 32 * 3  # After 2 max pooling layers
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Reshape input for 1D convolution (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=2, dropout_prob=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Use bidirectional LSTM
        )
        self.bn = nn.BatchNorm1d(hidden_size * 2)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take last sequence output
        x = self.bn(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class EnsembleModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=64):
        super().__init__()
        self.cnn = CNN(input_size=input_size, hidden_size=hidden_size)
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size)
        
        # Initialize weights with a slight bias towards positive predictions
        self.weight_cnn = nn.Parameter(torch.tensor([0.6]))
        self.weight_lstm = nn.Parameter(torch.tensor([0.4]))
        
        # Add a small bias term
        self.bias = nn.Parameter(torch.tensor([0.001]))
    
    def forward(self, x, return_individual=False):
        cnn_pred = self.cnn(x)
        lstm_pred = self.lstm(x)
        
        # Normalize weights using softmax
        weights = F.softmax(torch.stack([self.weight_cnn, self.weight_lstm]), dim=0)
        ensemble_pred = weights[0] * cnn_pred + weights[1] * lstm_pred + self.bias
        
        if return_individual:
            return ensemble_pred, cnn_pred, lstm_pred, weights.detach().numpy()
        return ensemble_pred
