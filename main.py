import warnings
warnings.filterwarnings('ignore')

import torch
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models import EnsembleModel

def load_model():
    # Initialize ensemble model
    model = EnsembleModel(
        cnn_hidden_size1=128, 
        cnn_hidden_size2=64,
        lstm_input_size=5,  # [Open, High, Low, Close, Volume]
        lstm_hidden_size=64,
        dropout_prob=0.3
    )
    
    # Load pre-trained models
    model.load_cnn_model('stock_prediction_model.pth')
    model.load_lstm_model('lstm_stock_model.pth')
    
    # Load the scaler
    scaler = joblib.load('stock_scaler.pkl')
    
    return model, scaler

def get_stock_data(symbol="RELIANCE.NS", days=20):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), 
                      end=end_date.strftime('%Y-%m-%d'), interval="1h")
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

def prepare_data(data, scaler, sequence_length=15):
    # Scale the data
    scaled_data = scaler.transform(data)
    
    # Create sequences
    sequences = []
    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:i + sequence_length])
    
    return torch.FloatTensor(sequences)

def predict_next_price(model, data, scaler):
    model.eval()
    with torch.no_grad():
        predictions, cnn_pred, lstm_pred, weights = model(data, return_individual=True)
        
        # Function to convert model output to actual price
        def to_price(pred):
            return scaler.inverse_transform(
                np.concatenate([np.zeros((pred.shape[0], 3)), 
                              pred.numpy(), 
                              np.zeros((pred.shape[0], 1))], axis=1)
            )[:, 3][-1]
        
        # Get predictions for all models
        ensemble_price = to_price(predictions)
        cnn_price = to_price(cnn_pred)
        lstm_price = to_price(lstm_pred)
        
        return ensemble_price, cnn_price, lstm_price, weights.numpy()

def main():
    # Load model and scaler
    model, scaler = load_model()
    
    # Get recent stock data
    data = get_stock_data()
    
    # Prepare data for prediction
    sequences = prepare_data(data, scaler)
    
    # Make predictions
    ensemble_price, cnn_price, lstm_price, weights = predict_next_price(model, sequences, scaler)
    current_price = float(data['Close'].iloc[-1])
    
    # Print results
    print("\nModel Predictions:")
    print(f"Current Close Price: {current_price:.2f}")
    print(f"\nCNN Model (weight: {weights[0]:.2f}):")
    print(f"Predicted Price: {cnn_price:.2f}")
    print(f"Predicted Change: {((cnn_price - current_price) / current_price * 100):.2f}%")
    
    print(f"\nLSTM Model (weight: {weights[1]:.2f}):")
    print(f"Predicted Price: {lstm_price:.2f}")
    print(f"Predicted Change: {((lstm_price - current_price) / current_price * 100):.2f}%")
    
    print(f"\nEnsemble Model (weighted average):")
    print(f"Predicted Price: {ensemble_price:.2f}")
    print(f"Predicted Change: {((ensemble_price - current_price) / current_price * 100):.2f}%")

if __name__ == "__main__":
    main()