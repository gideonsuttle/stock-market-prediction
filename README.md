# Stock Market Price Prediction using Deep Learning

This project implements deep learning models (CNN and LSTM) to predict stock market prices using historical data. The models are trained on stock data from the National Stock Exchange of India (NSE) using the RELIANCE stock as an example.

## Features

- Real-time stock data fetching using yfinance
- Data preprocessing and normalization
- Implementation of both CNN and LSTM models
- Model saving and loading functionality
- Real-time price predictions
- Performance metrics calculation

## Models Implemented

### 1. CNN Model
- Uses 1D convolutional layers for time series analysis
- Includes dropout layers for preventing overfitting
- Optimized using SGD optimizer

### 2. LSTM Model
- Processes sequential data using LSTM layers
- Includes dropout for regularization
- Uses Adam optimizer for training


## Usage

1. Clone the repository:

git clone https://github.com/gideonsuttle/stock-market-prediction.git
cd stock-market-prediction

2. Install dependencies:
bash
pip install -r requirements.txt

3. Run the Jupyter notebooks:
- `cnn_stock_market.ipynb` for CNN implementation
- `lstm_stock_market.ipynb` for LSTM implementation

## Model Training

The models are trained on historical stock data with the following parameters:
- Sequence length: 15 time steps
- Features: Open, High, Low, Close, Volume
- Training period: 50 days
- Interval: 30 minutes

## Prediction

The models can predict stock prices for the next time step based on the previous 15 time steps of data. Example usage:

python
Load saved model
model, scaler = load_model()
Get prediction
predicted_price = predict_stock_price(model, scaler, new_data)

## Results

- The models achieve competitive prediction accuracy
- Real-time prediction capability
- Error metrics (RMSE, MAE) are used for model evaluation

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)