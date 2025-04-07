import warnings
warnings.filterwarnings('ignore')

import torch
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models import EnsembleModel
import os
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries
from pathlib import Path
import json

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
if not API_KEY:
    raise ValueError("Please set your Alpha Vantage API key in the .env file")

# Create cache directory if it doesn't exist
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

def load_model():
    """Load the trained model and scaler"""
    # Load the model directly
    model = torch.load('models/ensemble_model.pth', map_location=torch.device('cpu'))
    model.eval()
    
    # Load the scaler
    scaler = joblib.load('models/scaler.pkl')
    
    return model, scaler

def get_cached_data(symbol):
    """Check if we have valid cached data"""
    cache_file = CACHE_DIR / f"{symbol}_daily.parquet"
    metadata_file = CACHE_DIR / f"{symbol}_daily_metadata.json"
    
    if not cache_file.exists() or not metadata_file.exists():
        return None
        
    # Check if cache is recent (less than 1 day old for daily data)
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        cache_time = datetime.fromisoformat(metadata['timestamp'])
        if datetime.now() - cache_time > timedelta(days=1):
            return None
            
        # Read cached data
        data = pd.read_parquet(cache_file)
        return data
    except Exception:
        return None

def save_to_cache(symbol, data):
    """Save data to cache with metadata"""
    try:
        cache_file = CACHE_DIR / f"{symbol}_daily.parquet"
        metadata_file = CACHE_DIR / f"{symbol}_daily_metadata.json"
        
        # Save the data
        data.to_parquet(cache_file)
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
    except Exception as e:
        print(f"Warning: Failed to cache data: {e}")

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26):
    """Calculate MACD technical indicator"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

def get_stock_data(symbol="AAPL", days=60):
    """
    Fetch stock data using Alpha Vantage API
    Returns daily data for analysis
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Try to get cached data first
    cached_data = get_cached_data(symbol)
    if cached_data is not None:
        print("Using cached data")
        data = cached_data
        
        # Calculate technical indicators for cached data
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        data['MACD'] = calculate_macd(data['Close'])
        data = data.fillna(method='ffill')
    else:
        print("Fetching fresh data from Alpha Vantage")
        ts = TimeSeries(key=API_KEY, output_format='pandas')
        
        try:
            # Get daily data
            data, _ = ts.get_daily(symbol=symbol, outputsize='full')
            
            # Rename columns
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Filter date range
            data.index = pd.to_datetime(data.index)
            data = data[data.index >= start_date]
            
            # Calculate technical indicators
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['RSI'] = calculate_rsi(data['Close'])
            data['MACD'] = calculate_macd(data['Close'])
            
            # Forward fill NaN values
            data = data.fillna(method='ffill')
            
            # Save to cache
            save_to_cache(symbol, data)
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None
    
    return data

def prepare_data(data, scaler, sequence_length=15):
    """Prepare data for prediction"""
    # Scale the data
    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'MACD']])
    
    # Create sequence
    sequence = torch.FloatTensor(scaled_data[-sequence_length:]).unsqueeze(0)
    return sequence

def predict_next_price(model, data, scaler):
    """Get prediction for the next price"""
    sequence = prepare_data(data, scaler)
    
    with torch.no_grad():
        prediction = model(sequence)
        
    # Convert prediction back to original scale
    scaled_prediction = prediction.numpy()
    # Create a dummy array with the same shape as the input data
    dummy = np.zeros((1, 9))
    dummy[:, 3] = scaled_prediction  # Put the prediction in the Close price column
    original_scale_pred = scaler.inverse_transform(dummy)[0, 3]
    
    return original_scale_pred

def main():
    try:
        # Load model and scaler
        model, scaler = load_model()
        
        # Get stock data
        symbol = "AAPL"  # Using the same symbol we trained on
        data = get_stock_data(symbol)
        
        if data is None or len(data) == 0:
            print("No data available")
            return
            
        # Get prediction
        next_price = predict_next_price(model, data, scaler)
        
        # Get current price
        current_price = data['Close'].iloc[-1]
        
        # Calculate price change
        price_change = next_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        print(f"\nCurrent {symbol} price: ${current_price:.2f}")
        print(f"Predicted next price: ${next_price:.2f}")
        print(f"Predicted change: ${price_change:.2f} ({price_change_pct:.2f}%)")
        
        # Trading signal
        if price_change_pct > 1.0:
            print("\nSignal: BUY ")
            print("Reason: Model predicts significant price increase")
        elif price_change_pct < -1.0:
            print("\nSignal: SELL ")
            print("Reason: Model predicts significant price decrease")
        else:
            print("\nSignal: HOLD ")
            print("Reason: No significant price movement predicted")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()