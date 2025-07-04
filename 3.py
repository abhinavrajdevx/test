# ===== Clean Inference & Plotting for 20 random points ===== #
model.load_state_dict(torch.load("best_lstm_multi_step.pth"))
model.eval()

random_points = random.sample(range(seq_length, len(df) - future_steps - 1), 20)

with torch.no_grad():
    for idx, point in enumerate(random_points):
        # Use all available history for this point
        input_seq = df_scaled.iloc[:point].values
        input_seq_tensor = torch.tensor(input_seq.reshape(1, input_seq.shape[0], len(features)), dtype=torch.float32).to(device)

        # Predict
        pred_scaled = model(input_seq_tensor).cpu().numpy().flatten()

        # Inverse transform predictions
        dummy_preds = np.zeros((future_steps, len(features)))
        dummy_preds[:, 3] = pred_scaled
        preds_unscaled = scaler.inverse_transform(dummy_preds)[:, 3]

        # Get actual future close prices
        actual_future = df['Close'].iloc[point: point + future_steps].values

        # Get corresponding timestamps
        future_times = df['Open time'].iloc[point: point + future_steps]

        # Plot only the prediction window
        plt.figure(figsize=(12, 5))
        plt.plot(future_times, actual_future, label='Actual Close', linewidth=1.5, color='blue')
        plt.plot(future_times, preds_unscaled, '--', label='LSTM Prediction', linewidth=2, color='orange')
        plt.title(f'Multi-Step Prediction (Point {idx+1} | Index {point})')
        plt.xlabel('Time')
        plt.ylabel('BTCUSDT Close Price')
        plt.legend()
        plt.tight_layout()
        plt.show()
