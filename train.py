import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from models import CNN, LSTM, EnsembleModel
from alpha_vantage.timeseries import TimeSeries
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

def get_training_data(symbol="RELIANCE.BSE", start_date="2023-01-01", end_date="2024-01-31"):
    """Fetch and prepare training data"""
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    
    try:
        # Get daily data
        print(f"Fetching training data for {symbol}...")
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        
        # Rename columns
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert index to datetime
        data.index = pd.to_datetime(data.index)
        
        # Filter date range
        mask = (data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))
        data = data.loc[mask]
        
        # Calculate technical indicators
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        data['MACD'] = calculate_macd(data['Close'])
        
        # Forward fill NaN values
        data = data.fillna(method='ffill')
        
        print(f"Successfully fetched {len(data)} data points")
        return data
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    """Calculate MACD"""
    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal

def prepare_sequences(data, sequence_length=15):
    """Prepare sequences for training"""
    data = data.copy()  # Make a copy to avoid modifying original data
    
    # Calculate indicators
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'] = calculate_macd(data['Close'])
    
    # Normalize price-based features to percentage changes
    price_cols = ['Open', 'High', 'Low', 'Close', 'SMA_5', 'SMA_20']
    for col in price_cols:
        data[col] = data[col].pct_change()
    
    # Normalize volume
    data['Volume'] = (data['Volume'] - data['Volume'].mean()) / data['Volume'].std()
    
    # Normalize RSI to [-1, 1]
    data['RSI'] = (data['RSI'] - 50) / 50
    
    # MACD is already normalized by its nature
    
    # Forward fill NaN values
    data = data.fillna(0)  # Fill NaN with 0 for first day's returns
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'MACD']
    
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        sequence = data[features].iloc[i:i+sequence_length].values
        target = data['Close'].iloc[i+sequence_length]  # Next day's return
        
        # Skip if any NaN values
        if np.isnan(sequence).any() or np.isnan(target):
            continue
            
        sequences.append(sequence)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    """Train the model"""
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        n_train_batches = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            
            # Skip bad batches
            if torch.isnan(loss):
                continue
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            n_train_batches += 1
        
        model.eval()
        val_loss = 0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                
                # Skip bad batches
                if torch.isnan(loss):
                    continue
                    
                val_loss += loss.item()
                n_val_batches += 1
        
        # Calculate average losses
        train_loss = train_loss / n_train_batches if n_train_batches > 0 else float('nan')
        val_loss = val_loss / n_val_batches if n_val_batches > 0 else float('nan')
        
        if val_loss < best_val_loss and not np.isnan(val_loss):
            best_val_loss = val_loss
            torch.save(model, 'models/ensemble_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

def main():
    # Get training data
    data = get_training_data()
    if data is None:
        return
    
    # Prepare sequences
    sequences, targets = prepare_sequences(data)
    
    # Scale the data
    scaler = StandardScaler()
    sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
    sequences_scaled = scaler.fit_transform(sequences_reshaped)
    sequences = sequences_scaled.reshape(sequences.shape)
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Convert to tensors
    X = torch.FloatTensor(sequences)
    y = torch.FloatTensor(targets)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    input_size = sequences.shape[-1]
    model = EnsembleModel(input_size=input_size)
    
    # Training parameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("\nTraining model...")
    train_model(model, train_loader, val_loader, criterion, optimizer)
    print("Training completed!")

if __name__ == "__main__":
    main()
