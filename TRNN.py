import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import yahoo_fin.stock_info as si
import os


# Technical indicators functions
def calculate_obv(closes, volumes):
    obv = np.zeros_like(closes)
    obv[0] = 0

    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def calculate_ad_indicator(highs, lows, closes, volumes):
    ad_indicator = np.zeros_like(closes)

    for i in range(1, len(closes)):
        if highs[i] != lows[i]:
            mf_multiplier = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i])
            ad_volume = mf_multiplier * volumes[i]
            ad_indicator[i] = ad_indicator[i - 1] + ad_volume
        else:
            ad_indicator[i] = ad_indicator[i - 1]

    return ad_indicator


def calculate_macd(close, short_window=12, long_window=26, signal_window=9):
    close_series = pd.Series(close)
    ema_short = close_series.ewm(span=short_window, adjust=False).mean()
    ema_long = close_series.ewm(span=long_window, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    return macd_histogram.values


def calculate_rsi(close, window=14):
    delta = np.diff(close)
    gain = np.maximum(0, delta)
    loss = -np.minimum(0, delta)
    avg_gain = np.mean(gain[:window])
    avg_loss = np.mean(loss[:window])
    rsi_values = np.zeros_like(close)

    if avg_loss != 0:
        rs = avg_gain / avg_loss
        rsi_values[window] = 100 - (100 / (1 + rs))

    for i in range(window + 1, len(close)):
        avg_gain = ((window - 1) * avg_gain + gain[i - 1]) / window
        avg_loss = ((window - 1) * avg_loss + loss[i - 1]) / window

        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100 - (100 / (1 + rs))
        else:
            rsi_values[i] = 100

    return rsi_values


# Fetch data using yahoo_fin
data = si.get_data('AAPL')
data.sort_index(inplace=True)

# Preprocess data
scaler = MinMaxScaler(feature_range=(-1, 1))
price = scaler.fit_transform(data[['adjclose']].values.reshape(-1, 1))

close = data['close'].values
volume = data['volume'].values
low = data['low'].values
high = data['high'].values

obv_values = calculate_obv(close, volume)
ad_values = calculate_ad_indicator(high, low, close, volume)
macd_histogram = calculate_macd(close)
rsi_values = calculate_rsi(close)


# Split data
def split_data(stock, lookback):
    data_raw = stock.to_numpy()
    data = []
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])
    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size
    x_train = data[:train_set_size, :-1]
    y_train = data[:train_set_size, -1]
    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1]
    return x_train, y_train, x_test, y_test


lookback = 30
x_train, y_train, x_test, y_test = split_data(pd.DataFrame(price), lookback)

# Convert data to PyTorch tensors
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).float()


# Enhanced model with LSTM, dropout, and additional layers
class EnhancedTRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(EnhancedTRNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.fc1(out[:, -1, :])  # Using the last output of LSTM
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Check if a saved model exists
model_path = 'EnhancedTRNN_final.pth'
if os.path.exists(model_path):
    model = EnhancedTRNN(input_dim=1, hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load(model_path))
    print("Loaded pre-trained model.")
else:
    model = EnhancedTRNN(input_dim=1, hidden_dim=64, output_dim=1)
    print("No pre-trained model found. Creating a new model.")

# Use Huber loss
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
train_losses = []
test_losses = []
mean_error_percentage = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    train_losses.append(loss.item())
    loss.backward()
    optimizer.step()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())

        # Calculate mean error percentage
        test_predictions = scaler.inverse_transform(test_outputs.numpy().reshape(-1, 1))
        actuals = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
        mape = np.mean(np.abs((actuals - test_predictions) / actuals)) * 100
        mean_error_percentage.append(mape)

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, MAPE: {mape:.4f}')

# Save final trained model
torch.save(model.state_dict(), 'EnhancedTRNN_final.pth')

# Make predictions
model.eval()
with torch.no_grad():
    train_predictions = model(x_train)
    test_predictions = model(x_test)

# Invert predictions
train_predictions = scaler.inverse_transform(train_predictions.numpy().reshape(-1, 1))
y_train = scaler.inverse_transform(y_train.numpy().reshape(-1, 1))
test_predictions = scaler.inverse_transform(test_predictions.numpy().reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')

# Plot training and test losses
plt.figure(figsize=(15, 9))
plt.plot(train_losses, label='Train Loss', color='royalblue')
plt.plot(test_losses, label='Test Loss', color='tomato')
plt.title('Training and Test Losses', fontsize=18, fontweight='bold')
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.legend()
plt.show()

# Plot mean error percentage for each epoch
plt.figure(figsize=(15, 9))
plt.plot(mean_error_percentage, label='Mean Error Percentage', color='green')
plt.title('Mean Error Percentage for Each Epoch', fontsize=18, fontweight='bold')
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Mean Error Percentage', fontsize=18)
plt.legend()
plt.show()

# Plot training predictions
sns.set_style("darkgrid")
plt.figure(figsize=(15, 9))
plt.plot(y_train, label="Actual", color='royalblue')
plt.plot(train_predictions, label="Predicted", color='tomato')
plt.title("Training Predictions", fontsize=18, fontweight='bold')
plt.xlabel('Days', fontsize=18)
plt.ylabel('Close Price (USD)', fontsize=18)
plt.legend()
plt.show()

# Plot test predictions vs actual prices
plt.figure(figsize=(15, 9))
plt.plot(data.index[-len(y_test):], y_test, label="Actual", color='royalblue')
plt.plot(data.index[-len(y_test):], test_predictions, label="Predicted", color='tomato')
plt.title("Test Predictions vs Actual Prices", fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price (USD)', fontsize=18)
plt.legend()
plt.show()