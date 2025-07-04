import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== Fetch BTCUSDT Futures 5m Data from Binance ===== #
def fetch_binance_futures(symbol='BTCUSDT', interval='5m', total_limit=20000):
    df_all = pd.DataFrame()
    end_time = None
    fetched = 0
    limit_per_call = 1000

    while fetched < total_limit:
        url = 'https://fapi.binance.com/fapi/v1/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit_per_call
        }
        if end_time:
            params['endTime'] = end_time

        response = requests.get(url, params=params)
        data = response.json()
        if not data:
            break

        df = pd.DataFrame(data, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ])
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

        df_all = pd.concat([df, df_all], ignore_index=True)
        fetched += len(df)
        end_time = int(df['Open time'].iloc[0].timestamp() * 1000) - 1
        time.sleep(0.1)

    return df_all[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values('Open time').reset_index(drop=True)

df = fetch_binance_futures(total_limit=20000)

# ===== Feature Engineering ===== #
df['Return'] = df['Close'].pct_change().fillna(0)
df['Volatility'] = df['Return'].rolling(12).std().fillna(0)
df['MA_10'] = df['Close'].rolling(10).mean().fillna(method='bfill')
df['MA_50'] = df['Close'].rolling(50).mean().fillna(method='bfill')
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Volatility', 'MA_10', 'MA_50']

# ===== Scaling ===== #
train_split = int(0.85 * len(df))
scaler = MinMaxScaler()
scaler.fit(df[features].iloc[:train_split])
df_scaled = scaler.transform(df[features])
df_scaled = pd.DataFrame(df_scaled, columns=features)

# ===== Sequence Preparation ===== #
seq_length = 60
future_steps = 30
data_X = []
data_Y = []

for i in range(len(df_scaled) - seq_length - future_steps):
    data_X.append(df_scaled.iloc[i:i+seq_length].values)
    data_Y.append(df_scaled.iloc[i+seq_length:i+seq_length+future_steps, 3].values)  # Close column

data_X = np.array(data_X)
data_Y = np.array(data_Y)

train_size = int(0.85 * len(data_X))

X_train = torch.tensor(data_X[:train_size], dtype=torch.float32).to(device)
y_train = torch.tensor(data_Y[:train_size], dtype=torch.float32).to(device)

X_val = torch.tensor(data_X[train_size:], dtype=torch.float32).to(device)
y_val = torch.tensor(data_Y[train_size:], dtype=torch.float32).to(device)

# ===== Model ===== #
class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(MultiStepLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = MultiStepLSTM(input_size=len(features), hidden_dim=64, num_layers=3, output_dim=future_steps).to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=7)

# ===== Training ===== #
epochs = 300
best_loss = float('inf')
patience = 15
trigger_times = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.8f} | Val Loss: {val_loss.item():.8f}")

    if val_loss.item() < best_loss:
        best_loss = val_loss.item()
        trigger_times = 0
        torch.save(model.state_dict(), "best_lstm_multi_step.pth")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# ===== Load Best Model ===== #
model.load_state_dict(torch.load("best_lstm_multi_step.pth"))
model.eval()

# ===== Visualization ===== #
random_points = random.sample(range(seq_length, len(df) - future_steps - 1), 10)

with torch.no_grad():
    for idx, point in enumerate(random_points):
        input_seq = df_scaled.iloc[point - seq_length: point].values
        input_seq_tensor = torch.tensor(input_seq.reshape(1, seq_length, len(features)), dtype=torch.float32).to(device)

        pred_scaled = model(input_seq_tensor).cpu().numpy().flatten()

        dummy = np.zeros((future_steps, len(features)))
        dummy[:, 3] = pred_scaled
        preds_unscaled = scaler.inverse_transform(dummy)[:, 3]

        actual_times = df['Open time'].iloc[point - seq_length: point + future_steps].reset_index(drop=True)
        actual_close = df['Close'].iloc[point - seq_length: point + future_steps].reset_index(drop=True)

        pred_times = df['Open time'].iloc[point: point + future_steps].reset_index(drop=True)

        plt.figure(figsize=(12, 5))
        plt.plot(actual_times, actual_close, label='Actual Close Price', linewidth=1.5)
        plt.plot(pred_times, preds_unscaled, label='Multi-Step LSTM Prediction', linestyle='--', linewidth=2, color='orange')

        plt.title(f'Multi-Step 30-Step LSTM Prediction (Point {idx+1} | Index {point})')
        plt.xlabel('Time')
        plt.ylabel('BTCUSDT Close Price')
        plt.legend()
        plt.tight_layout()
        plt.show()
