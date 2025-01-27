import joblib

def load_model():
    # Initialize model architecture
    model = cnn_for_time_series(hidden_size1=128, hidden_size2=64, dropout_prob=0.3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
    
    # Load the saved model state
    checkpoint = torch.load('stock_prediction_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load the scaler
    scaler = joblib.load('stock_scaler.pkl')
    
    return model, scaler, optimizer

# Example usage:
model, scaler, optimizer = load_model()
model.eval()