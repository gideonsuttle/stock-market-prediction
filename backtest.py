import warnings
warnings.filterwarnings('ignore')

import torch
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models import EnsembleModel
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class SimpleBacktester:
    def __init__(self, symbol="RELIANCE.NS", sequence_length=15, initial_cash=100000.0):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.trades = []
        self.portfolio_values = []
        self.highest_price_since_entry = 0  # Track highest price since position entry
        self.trailing_stop_pct = 0.03  # 3% trailing stop
        
        # Load model and scaler
        self.model, self.scaler = self.load_model()
        self.model.eval()
        
    def load_model(self):
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
    
    def prepare_data(self, data):
        # Handle multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            # Select the required columns
            data = data.droplevel('Ticker', axis=1)
            data = data.rename(columns={
                'Price': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
        
        # Scale the data
        scaled_data = self.scaler.transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
        
        # Create sequences
        sequences = []
        for i in range(len(scaled_data) - self.sequence_length):
            sequences.append(scaled_data[i:i + self.sequence_length])
        
        if len(sequences) > 0:
            return torch.FloatTensor(sequences)
        return None
    
    def get_prediction(self, sequences):
        with torch.no_grad():
            ensemble_pred, cnn_pred, lstm_pred, weights = self.model(sequences[-1:], return_individual=True)
            
        # Get the predicted price
        predicted_price = float(self.scaler.inverse_transform(
            np.concatenate([np.zeros((1, 3)), 
                          ensemble_pred.numpy(), 
                          np.zeros((1, 1))], axis=1)
        )[0, 3])
        
        return predicted_price, weights
    
    def calculate_dynamic_threshold(self, returns, lookback=20):
        """Calculate dynamic threshold based on recent volatility"""
        if len(returns) < lookback:
            return 0.008  # Default 0.8% threshold if not enough data
        
        # Calculate rolling volatility
        recent_volatility = float(returns[-lookback:].std())
        
        # Base threshold
        base_threshold = 0.008  # 0.8%
        
        # Adjust threshold based on volatility
        # Higher volatility = higher threshold
        dynamic_threshold = base_threshold * (1 + recent_volatility / 0.02)  # 0.02 is a scaling factor
        
        # Cap the threshold
        return min(max(dynamic_threshold, 0.006), 0.015)  # Between 0.6% and 1.5%
    
    def calculate_metrics(self, actual_prices, predicted_prices):
        """Calculate performance metrics for the model predictions"""
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mae = mean_absolute_error(actual_prices, predicted_prices)
        return {
            'RMSE': rmse,
            'MAE': mae
        }
    
    def run_backtest(self, start_date, end_date):
        # Download data
        print(f"\nDownloading data for {self.symbol} from {start_date} to {end_date}")
        data = yf.download(self.symbol, start=start_date, end=end_date)
        print(f"Downloaded {len(data)} data points")
        
        if data.empty:
            raise ValueError(f"No data available for {self.symbol} between {start_date} and {end_date}")
        
        print("\nFirst few rows of data:")
        print(data.head())
        
        # Handle multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel('Ticker', axis=1)
            data = data.rename(columns={
                'Price': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
        
        # Calculate returns for dynamic threshold
        returns = pd.Series([float(x) for x in data['Close'].pct_change().fillna(0)])
        
        # Prepare sequences
        sequences = self.prepare_data(data)
        if sequences is None:
            raise ValueError("Not enough data points for sequence creation")
        
        print(f"\nCreated {len(sequences)} sequences")
        
        # Reset state
        self.cash = self.initial_cash
        self.position = 0
        self.trades = []
        self.portfolio_values = []
        self.highest_price_since_entry = 0
        
        # Lists to store actual and predicted prices
        actual_prices = []
        predicted_prices = []
        
        # Trading simulation
        for i in range(self.sequence_length, len(data)):
            current_price = float(data['Close'].iloc[i])
            current_date = data.index[i]
            
            # Update highest price if we have a position
            if self.position > 0:
                self.highest_price_since_entry = max(self.highest_price_since_entry, current_price)
                
            # Get prediction
            seq = sequences[i - self.sequence_length]
            predicted_price, weights = self.get_prediction(seq.unsqueeze(0))
            
            actual_prices.append(current_price)
            predicted_prices.append(predicted_price)
            
            # Calculate predicted return
            predicted_return = (predicted_price - current_price) / current_price
            
            # Calculate dynamic threshold
            threshold = self.calculate_dynamic_threshold(returns[:i])
            
            # Print prediction info every 10 steps
            if i % 10 == 0:
                print(f"\nDate: {current_date}")
                print(f"Current Price: {current_price:.2f}")
                print(f"Predicted Price: {predicted_price:.2f}")
                print(f"Predicted Return: {predicted_return:.2%}")
                print(f"Dynamic Threshold: {threshold:.2%}")
                if self.position > 0:
                    print(f"Highest Price Since Entry: {self.highest_price_since_entry:.2f}")
                    print(f"Trailing Stop Price: {self.highest_price_since_entry * (1 - self.trailing_stop_pct):.2f}")
                print(f"CNN Weight: {weights[0]:.2f}, LSTM Weight: {weights[1]:.2f}")
            
            # Trading logic with trailing stop-loss
            if self.position > 0:
                # Calculate trailing stop price
                stop_price = self.highest_price_since_entry * (1 - self.trailing_stop_pct)
                
                # Check if we should sell based on trailing stop or prediction
                should_sell = (current_price <= stop_price) or \
                            (predicted_return < -threshold and current_price > stop_price)
                
                if should_sell:
                    # Sell signal
                    value = self.position * current_price
                    self.cash += value
                    position_return = (current_price - self.trades[-1]['price']) / self.trades[-1]['price']
                    self.trades.append({
                        'date': current_date,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': self.position,
                        'value': value
                    })
                    print(f"\nSELL: {self.position} shares at {current_price:.2f}")
                    print(f"Predicted Return: {predicted_return:.2%}")
                    print(f"Position Return: {position_return:.2%}")
                    print(f"Highest Price Reached: {self.highest_price_since_entry:.2f}")
                    print(f"Dynamic Threshold: {threshold:.2%}")
                    self.position = 0
                    self.highest_price_since_entry = 0
            
            elif predicted_return > threshold:
                # Buy signal
                position_size = 0.95  # Use 95% of available cash to leave room for fees
                shares = int((self.cash * position_size) // current_price)
                if shares > 0:  # Only buy if we can afford at least 1 share
                    cost = shares * current_price
                    self.cash -= cost
                    self.position = shares
                    self.highest_price_since_entry = current_price  # Initialize highest price
                    self.trades.append({
                        'date': current_date,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares,
                        'value': cost
                    })
                    print(f"\nBUY: {shares} shares at {current_price:.2f}")
                    print(f"Predicted Return: {predicted_return:.2%}")
                    print(f"Dynamic Threshold: {threshold:.2%}")
            
            # Record portfolio value
            portfolio_value = self.cash + (self.position * current_price)
            self.portfolio_values.append({
                'date': current_date,
                'value': portfolio_value
            })
        
        # Calculate metrics
        portfolio_values_df = pd.DataFrame(self.portfolio_values)
        returns = portfolio_values_df['value'].pct_change()
        
        final_value = portfolio_values_df['value'].iloc[-1]
        metrics = {
            'Total Return': (final_value - self.initial_cash) / self.initial_cash * 100,
            'Sharpe Ratio': np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0,
            'Max Drawdown': (portfolio_values_df['value'] / portfolio_values_df['value'].expanding(min_periods=1).max() - 1).min() * 100,
            'Final Portfolio Value': final_value,
            'Number of Trades': len(self.trades)
        }
        
        # Calculate performance metrics
        performance_metrics = self.calculate_metrics(actual_prices, predicted_prices)
        
        print("\nModel Performance Metrics:")
        print(f"Root Mean Squared Error (RMSE): {performance_metrics['RMSE']:.2f}")
        print(f"Mean Absolute Error (MAE): {performance_metrics['MAE']:.2f}")
        
        # Plot results
        self.plot_results(data, portfolio_values_df)
        
        return metrics
    
    def plot_results(self, price_data, portfolio_values_df):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Stock price and trade points
        ax1.plot(price_data.index, price_data['Close'], label='Stock Price', alpha=0.7)
        
        # Plot buy/sell points
        for trade in self.trades:
            if trade['action'] == 'BUY':
                ax1.scatter(trade['date'], trade['price'], color='green', marker='^', s=100)
            else:
                ax1.scatter(trade['date'], trade['price'], color='red', marker='v', s=100)
        
        ax1.set_title(f'{self.symbol} Stock Price and Trades')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Portfolio value
        ax2.plot(portfolio_values_df['date'], portfolio_values_df['value'], label='Portfolio Value')
        ax2.set_title('Portfolio Value Over Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def run_backtest(symbol="RELIANCE.NS", start_date="2023-12-01", end_date="2024-01-31", 
                initial_cash=100000.0):
    backtester = SimpleBacktester(symbol=symbol, initial_cash=initial_cash)
    return backtester.run_backtest(start_date, end_date)

if __name__ == "__main__":
    # Run backtest for a recent historical period
    metrics = run_backtest(
        symbol="RELIANCE.NS",
        start_date="2023-07-01",  # Extended to 6 months of data
        end_date="2024-02-13"
    )
    
    print("\nBacktest Results:")
    print("=================")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
