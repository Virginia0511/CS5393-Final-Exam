import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fetch stock data
def fetch_stock_data(ticker="AAPL", start_date="2010-01-01", end_date="2025-04-28"):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Prepare data with multiple features
def prepare_data(data, features=['Close', 'Volume']):
    df = data[features].copy()

    # Create additional volume features
    if 'Volume' in features:
        # Calculate volume change percentage
        df['Volume_Change'] = df['Volume'].pct_change()
        # Replace NaN with 0
        df['Volume_Change'] = df['Volume_Change'].fillna(0)
        features.append('Volume_Change')

    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    return scaled_data, scaler, features

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])  # Predicting the closing price
    return np.array(X), np.array(y)

# Build GRU model
def build_gru_model(seq_length, n_features, units=50, activation='tanh'):
    model = Sequential([
        GRU(units, return_sequences=True,
            input_shape=(seq_length, n_features),
            activation=activation),
        GRU(units, return_sequences=False,
            activation=activation),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Main function to run the prediction
def predict_stock_prices(ticker="AAPL", seq_length=60, epochs=100, batch_size=32):
    # Fetch data
    data = fetch_stock_data(ticker)

    # Prepare data
    features = ['Close', 'Volume']
    scaled_data, scaler, features = prepare_data(data, features)

    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)

    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build model
    model = build_gru_model(seq_length, len(features))

    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )

    # Make predictions
    predictions = model.predict(X_test)

    # Inverse transform predictions and actual values
    # Create arrays with zeros for features that are not predicted
    pred_array = np.zeros((len(predictions), len(features)))
    pred_array[:, 0] = predictions.flatten()

    y_test_array = np.zeros((len(y_test), len(features)))
    y_test_array[:, 0] = y_test

    # Inverse transform
    predictions_transformed = scaler.inverse_transform(pred_array)[:, 0]
    y_test_transformed = scaler.inverse_transform(y_test_array)[:, 0]

    # Calculate RMSE
    rmse = np.sqrt(np.mean((predictions_transformed - y_test_transformed) ** 2))
    print(f"RMSE: {rmse}")

    # Plot results
    plot_results(data, predictions_transformed, y_test_transformed, train_size, seq_length)

    return model, history, predictions_transformed, y_test_transformed, rmse

# Function to plot results
def plot_results(data, predictions, actual, train_size, seq_length):
    plt.figure(figsize=(16, 8))

    # Get dates for test data
    test_dates = data.index[train_size + seq_length:]

    # Create a dataframe with predictions and actual values
    results_df = pd.DataFrame({
        'Actual': actual,
        'Predicted': predictions
    }, index=test_dates)

    # Plot
    plt.plot(results_df.index, results_df['Actual'], label='Actual Prices')
    plt.plot(results_df.index, results_df['Predicted'], label='Predicted Prices')

    plt.title('Stock Price Prediction with GRU Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('resultVIX.png')

# Function to experiment with different activation functions
def experiment_activations(ticker="AAPL", seq_length=60, epochs=50, batch_size=32):
    # List of activation functions
    activations = ['tanh', 'relu', 'sigmoid']
    results = {}

    for activation in activations:
        print(f"\nTraining model with activation function: {activation}")
        _, _, _, _, rmse = predict_stock_prices(
            ticker=ticker,
            seq_length=seq_length,
            epochs=epochs,
            batch_size=batch_size
        )
        results[activation] = rmse

    # Print summary of results
    print("\nResults Summary:")
    for activation, rmse in results.items():
        print(f"Activation: {activation}, RMSE: {rmse}")

    # Determine best activation
    best_activation = min(results, key=results.get)
    print(f"\nBest activation function: {best_activation} with RMSE: {results[best_activation]}")

# Run the prediction
if __name__ == "__main__":
    print("Stock Price Prediction with GRU and Volume Data")
    #model, history, predictions, actual, rmse = predict_stock_prices(ticker="AAPL")
    #model, history, predictions, actual, rmse = predict_stock_prices(ticker="TSLA")
    #model, history, predictions, actual, rmse = predict_stock_prices(ticker='SPY')
    #model, history, predictions, actual, rmse = predict_stock_prices(ticker="VTI")
    #model, history, predictions, actual, rmse = predict_stock_prices(ticker="^VIX")
    #model, history, predictions, actual, rsme = predict_stock_prices(ticker='NVDA')


    #experiment_activations(ticker='AAPL', epochs=50)
    #experiment_activations(ticker='TSLA', epochs=50)
    #experiment_activations(ticker='SPY', epochs=50)
    #experiment_activations(ticker='VTI', epochs=50)
    #experiment_activations(ticker='VIX', epochs=50)
    experiment_activations(ticker='NVDA',epochs=30)
    experiment_activations(ticker='NVDA', epochs=50)
