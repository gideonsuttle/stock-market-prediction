import torch
import torch.nn as nn
import math

class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, peephole=False, dropout_prob=0.1):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.peephole = peephole
        self.U = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.W = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.dropout_prob = dropout_prob
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
        self.dropout = nn.Dropout(dropout_prob)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(bs, self.hidden_size).to(x.device)
            c_t = torch.zeros(bs, self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            x_t = x[:, t, :]

            # Get all 4 gates using one big matrix multiplication
            gates = x_t @ self.U + h_t @ self.W + self.bias

            # Split gates
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :self.hidden_size]),                # input
                torch.sigmoid(gates[:, self.hidden_size:self.hidden_size*2]),  # forget
                torch.tanh(gates[:, self.hidden_size*2:self.hidden_size*3]),   # cell
                torch.sigmoid(gates[:, self.hidden_size*3:]),             # output
            )

            if self.peephole:
                i_t = i_t * c_t
                f_t = f_t * c_t

            c_t = f_t * c_t + i_t * g_t

            if self.peephole:
                o_t = o_t * c_t

            h_t = o_t * torch.tanh(c_t)
            h_t = self.dropout(h_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)

class cnn_for_time_series(nn.Module):
    def __init__(self, hidden_size1=128, hidden_size2=64, dropout_prob=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_size1, kernel_size=(3,3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=1)
        self.drop = nn.Dropout(p=dropout_prob)
        
        self.conv2 = nn.Conv2d(in_channels=hidden_size1, out_channels=hidden_size2, kernel_size=(3,3), stride=1)
        
        self.fc1 = nn.Linear(hidden_size2 * 12 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.drop(x)
        
        x = self.conv2(x)
        x = self.drop(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

class EnsembleModel(nn.Module):
    def __init__(self, cnn_hidden_size1=128, cnn_hidden_size2=64, lstm_input_size=5, lstm_hidden_size=64, dropout_prob=0.3):
        super().__init__()
        self.cnn = cnn_for_time_series(hidden_size1=cnn_hidden_size1, hidden_size2=cnn_hidden_size2, dropout_prob=dropout_prob)
        self.lstm = CustomLSTM(input_sz=lstm_input_size, hidden_sz=lstm_hidden_size, dropout_prob=dropout_prob)
        
        # Additional layer to convert LSTM output to prediction
        self.lstm_out = nn.Linear(lstm_hidden_size, 1)
        
        # Weighted average of predictions
        self.cnn_weight = nn.Parameter(torch.tensor(0.5))
        self.lstm_weight = nn.Parameter(torch.tensor(0.5))
        
    def load_cnn_model(self, cnn_path):
        checkpoint = torch.load(cnn_path, map_location=torch.device('cpu'))
        self.cnn.load_state_dict(checkpoint['model_state_dict'])
        
    def load_lstm_model(self, lstm_path):
        checkpoint = torch.load(lstm_path, map_location=torch.device('cpu'))
        self.lstm.load_state_dict(checkpoint['model_state_dict'])
    
    def forward(self, x, return_individual=False):
        # Prepare input for CNN (add channel dimension)
        cnn_input = x.unsqueeze(1)  # Shape: [batch_size, 1, seq_len, features]
        cnn_pred = self.cnn(cnn_input)
        
        # Prepare input for LSTM
        lstm_seq, _ = self.lstm(x)  # Shape: [batch_size, seq_len, hidden_size]
        lstm_last = lstm_seq[:, -1, :]  # Take last timestep
        lstm_pred = self.lstm_out(lstm_last)  # Convert to prediction
        
        # Normalize weights using softmax
        weights = torch.softmax(torch.tensor([self.cnn_weight, self.lstm_weight]), dim=0)
        
        # Weighted average of predictions
        ensemble_pred = weights[0] * cnn_pred + weights[1] * lstm_pred
        
        if return_individual:
            return ensemble_pred, cnn_pred, lstm_pred, weights
        return ensemble_pred
