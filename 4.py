import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time

# Download historical data using requests
def download_data():
    end_time = int(time.time() * 1000)
    start_time = end_time - 10 * 24 * 60 * 60 * 1000  # 10 days in ms
    
    url = "https://fapi.binance.com/fapi/v1/klines"
    all_data = []
    
    while start_time < end_time:
        params = {
            'symbol': 'BTCUSDT',
            'interval': '5m',
            'limit': 1500,  # Max per request
            'startTime': start_time,
            'endTime': end_time
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        
        # Move start_time to last candle + 5m
        start_time = data[-1][0] + 300000  # 5m in ms
    
    # Create DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
               'taker_buy_quote', 'ignore']
    
    df = pd.DataFrame(all_data, columns=columns)
    
    # Convert to numeric types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'trades']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp').drop_duplicates()

# Preprocess data
def preprocess_data(df):
    # Feature engineering
    df['returns'] = df['close'].pct_change()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_ret'].rolling(12).std()
    
    # Technical indicators
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    
    # Normalize features
    features = ['close', 'volume', 'volatility', 'ma10', 'ma50', 'rsi']
    df = df[features].dropna()
    
    # Min-Max Scaling
    data = df.values
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    scaled_data = (data - data_min) / (data_max - data_min + 1e-8)
    
    return scaled_data, data_min, data_max

# Compute RSI indicator
def compute_rsi(prices, window=14):
    deltas = np.diff(prices)
    seed = deltas[:window]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100. / (1. + rs)

    for i in range(window, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi

# Create dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, pred_length):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length
        self.pred_length = pred_length
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length
        
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length:idx+self.seq_length+self.pred_length, 0]  # Predict close price
        return x, y

# Transformer Model
class PricePredictor(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=False,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, output_dim)
        self.pred_length = None
        
    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer(src)
        output = self.decoder(output)
        return output[-self.pred_length:]  # Return only prediction part

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Configuration
SEQ_LENGTH = 288  # 24 hours (288 * 5min)
PRED_LENGTH = 30  # 150 minutes (2.5 hours)
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0005
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DROPOUT = 0.2

# Main pipeline
if __name__ == "__main__":
    print("Downloading data...")
    df = download_data()
    print(f"Retrieved {len(df)} data points")
    
    print("Preprocessing data...")
    data, data_min, data_max = preprocess_data(df)
    
    # Create dataset
    print("Creating datasets...")
    dataset = TimeSeriesDataset(data, SEQ_LENGTH, PRED_LENGTH)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = PricePredictor(
        input_dim=data.shape[1],
        output_dim=1,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    model.pred_length = PRED_LENGTH
    
    # Model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        start_time = time.time()
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Transformer expects (seq_len, batch, features)
            output = model(X.permute(1, 0, 2))
            loss = criterion(output.squeeze(-1), y.permute(1, 0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X.permute(1, 0, 2))
                loss = criterion(output.squeeze(-1), y.permute(1, 0))
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.1f}s | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Make final prediction
    model.eval()
    with torch.no_grad():
        last_sequence = torch.tensor(data[-SEQ_LENGTH:], dtype=torch.float32).unsqueeze(0).to(device)
        prediction = model(last_sequence.permute(1, 0, 2)).squeeze().cpu().numpy()
    
    # Inverse transform prediction
    close_min = data_min[0]
    close_max = data_max[0]
    prediction = prediction * (close_max - close_min) + close_min
    
    # Generate future timestamps
    last_time = df.index[-1]
    future_times = [last_time + timedelta(minutes=5*(i+1)) for i in range(PRED_LENGTH)]
    
    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(df.index[-200:], df['close'][-200:], label='Historical Price', linewidth=2)
    plt.plot(future_times, prediction, label='Predicted Price', marker='o', linestyle='-', color='orange')
    
    # Formatting
    plt.title('BTCUSDT 5min Price Prediction', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save and show
    plt.savefig('btc_prediction.png', dpi=300)
    plt.show()
    
    # Print prediction summary
    print("\nPrediction Summary:")
    print(f"Current Price: {df['close'].iloc[-1]:.2f}")
    print(f"Predicted High: {max(prediction):.2f}")
    print(f"Predicted Low: {min(prediction):.2f}")
