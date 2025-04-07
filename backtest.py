import warnings
warnings.filterwarnings('ignore')

import torch
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models import EnsembleModel
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from dotenv import load_dotenv
from pathlib import Path
import json
import pickle
import time

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
if not API_KEY:
    raise ValueError("Please set your Alpha Vantage API key in the .env file")

def get_cached_data(symbol, cache_key):
    """Get data from cache if available and not expired"""
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"{symbol}_{cache_key}.pkl")
    if os.path.exists(cache_file):
        cache_time = os.path.getmtime(cache_file)
        if time.time() - cache_time < 3600:  # Cache valid for 1 hour
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
    return None

def save_to_cache(symbol, data, cache_key):
    """Save data to cache"""
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"{symbol}_{cache_key}.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

def get_historical_data(symbol, start_date=None, end_date=None):
    """Get historical data from Alpha Vantage"""
    cache_file = f'cache/{symbol}_daily.csv'
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        data = pd.read_csv(cache_file, index_col='date', parse_dates=['date'])
        if len(data) > 100:  # If we have enough data in cache
            print("\nUsing cached data")
            return data
    
    print("Fetching fresh data from Alpha Vantage")
    try:
        # Initialize Alpha Vantage API
        ts = TimeSeries(key=API_KEY, output_format='pandas')
        
        # Get daily data
        print(f"Fetching data for symbol: {symbol}")
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        
        # Rename columns to match our format
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Sort by date
        data.sort_index(ascending=True, inplace=True)
        
        # Filter by date range if provided
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        print(f"\nSuccessfully fetched {len(data)} data points")
        print("\nFirst few rows:")
        print(data.head())
        
        # Save to cache
        os.makedirs('cache', exist_ok=True)
        data.to_csv(cache_file)
        
        return data
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        if os.path.exists(cache_file):
            print("Using older cached data due to error")
            return pd.read_csv(cache_file, index_col='date', parse_dates=['date'])
        return None

def calculate_rsi(prices):
    """Calculate Relative Strength Index (RSI)"""
    delta = prices.diff()
    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = abs(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    """Calculate Moving Average Convergence Divergence (MACD)"""
    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal

def prepare_backtest_data(data, sequence_length=15):
    """Prepare data for backtesting"""
    # Calculate technical indicators
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
    
    if len(data) < sequence_length:
        return None
        
    # Create sequence
    sequence = data[features].values[-sequence_length:]
    
    # Reshape for model (batch_size, sequence_length, n_features)
    sequence = sequence.reshape(1, sequence_length, len(features))
    return torch.FloatTensor(sequence)

class SimpleBacktester:
    def __init__(self, symbol="AAPL", sequence_length=15, initial_cash=100000.0):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.shares = 0
        self.trades = []
        self.portfolio_values = []
        
        # Load the trained model
        self.model = torch.load('models/ensemble_model.pth')
        self.model.eval()
        
        # Trading parameters - even more aggressive for limited data
        self.buy_threshold = -0.045  # Buy if predicted drop is less than 4.5%
        self.sell_threshold = -0.065  # Sell if predicted drop is more than 6.5%
        self.stop_loss = -0.01  # 1% stop loss
        self.take_profit = 0.015  # 1.5% take profit
    
    def plot_results(self, data):
        """Plot backtest results"""
        plt.figure(figsize=(15, 10))
        
        # Convert portfolio values to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df.set_index('Date', inplace=True)
        
        # Plot stock price
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['Close'], label='Stock Price', color='blue', alpha=0.6)
        plt.title('Stock Price and Trading Activity')
        plt.ylabel('Price ($)')
        
        # Plot buy/sell points
        for trade in self.trades:
            if trade['Action'] == 'BUY':
                plt.scatter(trade['Date'], trade['Price'], color='green', marker='^', s=100, label='Buy')
            else:
                plt.scatter(trade['Date'], trade['Price'], color='red', marker='v', s=100, label='Sell')
        
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.grid(True)
        
        # Plot portfolio value
        plt.subplot(2, 1, 2)
        plt.plot(portfolio_df.index, portfolio_df['Portfolio_Value'], label='Portfolio Value', color='orange')
        plt.title('Portfolio Value Over Time')
        plt.ylabel('Value ($)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/backtest_results.png')
        print("\nPlot saved as plots/backtest_results.png")
        plt.close()

    def run_backtest(self, start_date=None, end_date=None):
        """Run backtest simulation"""
        print("\nRunning backtest...")
        
        # Get historical data
        data = get_historical_data(self.symbol, start_date, end_date)
        if data is None or len(data) < self.sequence_length:
            raise ValueError("Not enough data points for backtesting")
        
        # Initialize tracking variables
        self.cash = self.initial_cash
        self.shares = 0
        self.trades = []
        self.portfolio_values = []
        
        print(f"\nStarting backtest with {len(data)} data points")
        print(f"Initial cash: ${self.initial_cash:,.2f}")
        print(f"Trading parameters:")
        print(f"- Buy threshold: {self.buy_threshold*100:.3f}%")
        print(f"- Sell threshold: {self.sell_threshold*100:.3f}%")
        print(f"- Stop loss: {self.stop_loss*100:.1f}%")
        print(f"- Take profit: {self.take_profit*100:.1f}%")
        
        # Iterate through each day
        for i in range(self.sequence_length, len(data)):
            current_data = data.iloc[:i]
            current_price = current_data['Close'].iloc[-1]
            
            # Prepare data and get prediction
            sequence = prepare_backtest_data(current_data, self.sequence_length)
            if sequence is None:
                print(f"Warning: Could not prepare sequence at index {i}")
                continue
                
            with torch.no_grad():
                predicted_return = self.model(sequence).item()
        
            print(f"\nDate: {current_data.index[-1]}")
            print(f"Current price: ${current_price:.2f}")
            print(f"Predicted return: {predicted_return*100:.3f}%")
            
            # Update portfolio value
            portfolio_value = self.cash + self.shares * current_price
            self.portfolio_values.append({
                'Date': current_data.index[-1],
                'Portfolio_Value': portfolio_value,
                'Shares': self.shares,
                'Cash': self.cash,
                'Predicted_Return': predicted_return
            })
            
            # Trading logic
            if self.shares == 0:  # No position
                # Buy when predicted drop is less severe than buy threshold
                if predicted_return > self.buy_threshold:
                    # Calculate position size (invest 95% of cash)
                    position_size = int((self.cash * 0.95) / current_price)
                    cost = position_size * current_price
                    
                    if position_size > 0:
                        self.shares = position_size
                        self.cash -= cost
                        print(f"BUY: {position_size} shares at ${current_price:.2f}")
                        print(f"Reason: Predicted return {predicted_return*100:.3f}% > {self.buy_threshold*100:.3f}%")
                        self.trades.append({
                            'Date': current_data.index[-1],
                            'Action': 'BUY',
                            'Price': current_price,
                            'Quantity': position_size,
                            'Value': cost,
                            'Portfolio_Value': portfolio_value
                        })
        
            else:  # Have position
                # Check for stop loss or take profit
                entry_price = self.trades[-1]['Price']
                current_return = (current_price - entry_price) / entry_price
                
                # Sell when predicted drop is more severe than sell threshold
                # OR stop loss/take profit is hit
                if (predicted_return < self.sell_threshold or 
                    current_return <= self.stop_loss or 
                    current_return >= self.take_profit):
                    
                    # Sell everything
                    sale_value = self.shares * current_price
                    self.cash += sale_value
                    
                    print(f"SELL: {self.shares} shares at ${current_price:.2f}")
                    if predicted_return < self.sell_threshold:
                        print(f"Reason: Predicted return {predicted_return*100:.3f}% < {self.sell_threshold*100:.3f}%")
                    elif current_return <= self.stop_loss:
                        print(f"Reason: Stop loss hit ({current_return*100:.2f}%)")
                    else:
                        print(f"Reason: Take profit hit ({current_return*100:.2f}%)")
                    
                    self.trades.append({
                        'Date': current_data.index[-1],
                        'Action': 'SELL',
                        'Price': current_price,
                        'Quantity': self.shares,
                        'Value': sale_value,
                        'Portfolio_Value': portfolio_value
                    })
                    
                    self.shares = 0
    
        # Plot results
        self.plot_results(data)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        print("\nBacktest Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value:.2f}")
        
        return metrics
    
    def calculate_metrics(self):
        """Calculate backtest performance metrics"""
        if not self.portfolio_values:
            return {
                'Initial Value': self.initial_cash,
                'Final Value': self.cash,
                'Total Return': 0.0,
                'Total Trades': 0,
                'Win Rate': 0.0,
                'Sharpe Ratio': float('-inf'),
                'Max Drawdown': 0.0
            }
        
        # Convert portfolio values to DataFrame
        portfolio_values_df = pd.DataFrame(self.portfolio_values)
        portfolio_values_df.set_index('Date', inplace=True)
        
        # Calculate metrics
        initial_value = self.initial_cash
        final_value = portfolio_values_df['Portfolio_Value'].iloc[-1]
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        # Calculate daily returns
        portfolio_values_df['Daily_Return'] = portfolio_values_df['Portfolio_Value'].pct_change()
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        excess_returns = portfolio_values_df['Daily_Return'] - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(excess_returns) > 1 else 0
        
        # Calculate Maximum Drawdown
        portfolio_values_df['Peak'] = portfolio_values_df['Portfolio_Value'].expanding().max()
        portfolio_values_df['Drawdown'] = (portfolio_values_df['Portfolio_Value'] - portfolio_values_df['Peak']) / portfolio_values_df['Peak']
        max_drawdown = portfolio_values_df['Drawdown'].min() * 100
        
        # Calculate win rate
        profitable_trades = sum(1 for trade in self.trades if trade['Action'] == 'SELL' 
                              and trade['Value'] > trade['Quantity'] * self.trades[self.trades.index(trade)-1]['Price'])
        total_trades = len([trade for trade in self.trades if trade['Action'] == 'SELL'])
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'Initial Value': initial_value,
            'Final Value': final_value,
            'Total Return': total_return,
            'Total Trades': total_trades,
            'Win Rate': win_rate,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }
    
def run_backtest(symbol="AAPL", start_date="2023-12-01", end_date="2024-01-31", 
                initial_cash=100000.0):
    try:
        backtester = SimpleBacktester(symbol=symbol, initial_cash=initial_cash)
        metrics = backtester.run_backtest(start_date, end_date)
        
        print("\nBacktest Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")
            
        return metrics
    except Exception as e:
        print(f"\nBacktest failed: {str(e)}")
        return None

if __name__ == "__main__":
    # Run backtest for a recent historical period
    metrics = run_backtest(
        symbol="RELIANCE.BSE",  # Using Reliance from BSE
        start_date="2024-01-01",  # Using a shorter date range
        end_date="2024-01-31",
        initial_cash=100000.0
    )
