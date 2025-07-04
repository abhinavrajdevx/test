import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import ta  # Technical analysis library

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======= Fetch MAX Available BTCUSDT Futures Data ======= #
def fetch_max_binance_futures(symbol='BTCUSDT', interval='5m'):
    print("Fetching maximum available historical data...")
    df_all = pd.DataFrame()
    end_time = None
    limit_per_call = 1000  # Max allowed by Binance API
    call_count = 0
    max_calls = 500  # Safety limit to prevent infinite loops
    
    while call_count < max_calls:
        url = 'https://fapi.binance.com/fapi/v1/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit_per_call
        }
        if end_time:
            params['endTime'] = end_time

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print("No more data available.")
                break

            df = pd.DataFrame(data, columns=[
                'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close time', 'Quote asset volume', 'Number of trades',
                'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
            ])
            
            # Convert to proper types
            df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
            df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            
            # Add to main dataframe
            df_all = pd.concat([df, df_all], ignore_index=True)
            
            # Set next end time (first candle of current batch minus 1ms)
            end_time = int(df['Open time'].iloc[0].timestamp() * 1000) - 1
            call_count += 1
            
            print(f"Fetched {len(df)} candles | Total: {len(df_all)} | Oldest: {df['Open time'].iloc[0]}")
            
            # Sleep to avoid rate limits (Binance: 1200 requests/min)
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(1)
            continue

    return df_all[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values('Open time').reset_index(drop=True)

# Fetch MAX available data
df = fetch_max_binance_futures()
print(f"\nTotal data points: {len(df)}")
print(f"Date range: {df['Open time'].iloc[0]} to {df['Open time'].iloc[-1]}")

# ======= Feature Engineering ======= #
print("\nCalculating technical indicators...")
# Add technical indicators
df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['MACD_diff'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9).macd_diff()
df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()

# Drop rows with NaN values
initial_count = len(df)
df.dropna(inplace=True)
print(f"Dropped {initial_count - len(df)} rows with missing values")

# Enhanced feature set
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD_diff', 'SMA_20', 'EMA_50']

# ======= Data Preprocessing ======= #
print("\nPreprocessing data...")
# Split data before scaling to prevent leakage
train_size = int(0.9 * len(df))
train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:]

# Scale features
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df[features])
val_scaled = scaler.transform(val_df[features])

# Combine scaled data
all_scaled = np.vstack([train_scaled, val_scaled])
df_scaled = pd.DataFrame(all_scaled, columns=features)

seq_length = 90  # 7.5 hours of historical context
target_steps = 1  # Predict next step only

# Create sequences and targets
data = []
targets = []
for i in range(len(df_scaled) - seq_length - target_steps + 1):
    data.append(df_scaled.iloc[i:i+seq_length].values)
    targets.append(df_scaled.iloc[i+seq_length:i+seq_length+target_steps, 3].values)  # Close price

data = np.array(data)
targets = np.array(targets)

# Split into training and validation sets
train_end = train_size - seq_length - target_steps + 1
X_train = torch.tensor(data[:train_end], dtype=torch.float32).to(device)
y_train = torch.tensor(targets[:train_end], dtype=torch.float32).to(device)
X_val = torch.tensor(data[train_end:], dtype=torch.float32).to(device)
y_val = torch.tensor(targets[train_end:], dtype=torch.float32).to(device)

print(f"Total sequences: {len(data)}")
print(f"Training sequences: {len(X_train)}")
print(f"Validation sequences: {len(X_val)}")

# Create DataLoaders for batch training
batch_size = 256 if device.type == 'cuda' else 64
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ======= Enhanced LSTM Model ======= #
class PredictionModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(PredictionModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_dim, 
            num_layers, 
            dropout=dropout, 
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last timestep
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

model = PredictionModel(
    input_size=len(features), 
    hidden_dim=128, 
    num_layers=3, 
    output_dim=target_steps
).to(device)

# Use GPU if available
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    model = nn.DataParallel(model)  # Parallelize if multiple GPUs

criterion = nn.HuberLoss()  # Robust loss function
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=8
)

# ======= Training Loop ======= #
print("\nStarting training...")
epochs = 5
best_loss = float('inf')
patience = 15
trigger_times = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), "best_lstm_model.pth")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# ======= Recursive Predictions ======= #
model.load_state_dict(torch.load("best_lstm_model.pth"))
model.eval()

print("\nMaking predictions...")
# Use last 100 points for prediction visualization
prediction_points = list(range(len(df) - 100, len(df) - 30))

with torch.no_grad():
    for idx, point in enumerate(prediction_points[:5]):  # Show first 5
        # Initialize with real data
        input_seq = df_scaled.iloc[point-seq_length:point].values
        input_tensor = torch.tensor(input_seq[None, ...], dtype=torch.float32).to(device)
        
        preds_scaled = []
        current_seq = input_tensor.clone()
        
        # Recursive prediction for next 30 steps
        for step in range(30):
            pred = model(current_seq)
            pred_value = pred.cpu().numpy().flatten()[0]
            preds_scaled.append(pred_value)
            
            # Update sequence
            last_step = current_seq[:, -1:, :].clone()
            last_close = last_step[:, :, 3]
            
            # Create new candle
            new_step = last_step.clone()
            new_step[:, :, 0] = last_close  # Open = previous close
            new_step[:, :, 3] = pred        # Close = prediction
            new_step[:, :, 1] = torch.maximum(last_close, pred)  # High
            new_step[:, :, 2] = torch.minimum(last_close, pred)  # Low
            
            # Volume and indicators remain the same (simplified)
            current_seq = torch.cat((current_seq[:, 1:, :], new_step), dim=1)
        
        # Inverse transform predictions
        dummy = np.zeros((30, len(features)))
        dummy[:, 3] = preds_scaled
        preds_unscaled = scaler.inverse_transform(dummy)[:, 3]
        
        # Get actual values
        actual_range = df['Close'].iloc[point-30:point+30].values
        actual_times = df['Open time'].iloc[point-30:point+30]
        pred_times = df['Open time'].iloc[point:point+30]
        
        # Plot results
        plt.figure(figsize=(14, 7))
        plt.plot(actual_times, actual_range, 'b-', label='Actual Price')
        plt.plot(pred_times, preds_unscaled, 'ro-', label='30-Step Prediction')
        
        # Mark prediction start
        plt.axvline(x=actual_times.iloc[30], color='g', linestyle='--', 
                   label='Prediction Start')
        
        plt.title(f'BTCUSDT Price Prediction from {actual_times.iloc[30]}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Calculate metrics
        actuals = df['Close'].iloc[point:point+30].values
        rmse = np.sqrt(mean_squared_error(actuals, preds_unscaled))
        mae = np.mean(np.abs(actuals - preds_unscaled))
        
        print(f"Prediction {idx+1} from {actual_times.iloc[30]}:")
        print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        print(f"  First Step Error: {abs(actuals[0] - preds_unscaled[0]):.2f}")
        print(f"  Last Step Error: {abs(actuals[-1] - preds_unscaled[-1]):.2f}")
