import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def main():
    print("====================================================")
    print("Project 46: Temperature Forecasting with LSTM")
    print("Goal: Build an LSTM time-series model to predict daily temperatures")
    print("====================================================\n")

    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # 1. Generate Synthetic Daily Temperature Dataset over 3 Years (1095 days)
    num_days = 1095
    t = np.arange(num_days)
    
    # Components: Seasonality (Sine), warming trend, and random Gaussian noise
    seasonality = 18.0 + 12.0 * np.sin(2.0 * np.pi * t / 365.0 - np.pi / 2.0)
    trend = 0.002 * t
    noise = np.random.normal(0, 2.0, num_days)
    temperatures = seasonality + trend + noise

    print(f"Generated daily temperature series for {num_days} days (3 years).")
    print(f"  Min Temp: {np.min(temperatures):.2f} °C")
    print(f"  Max Temp: {np.max(temperatures):.2f} °C")
    print(f"  Mean Temp: {np.mean(temperatures):.2f} °C\n")

    # 2. Train-Test Split (80% Train, 20% Test)
    split_idx = int(num_days * 0.8)
    train_data = temperatures[:split_idx]
    test_data = temperatures[split_idx:]

    print(f"Data partitioning:")
    print(f"  Training samples: {len(train_data)} days")
    print(f"  Testing samples:  {len(test_data)} days\n")

    # 3. MinMax Scaling based on Training Set statistics (vital for LSTM)
    min_val = np.min(train_data)
    max_val = np.max(train_data)

    def scale(data):
        return (data - min_val) / (max_val - min_val)

    def descale(scaled_data):
        return scaled_data * (max_val - min_val) + min_val

    train_scaled = scale(train_data)
    test_scaled = scale(test_data)

    # 4. Create sequences using a sliding lookback window
    def create_sequences(data, lookback=30):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i : i + lookback])
            y.append(data[i + lookback])
        # Return reshaped X as [samples, time_steps, features]
        return np.expand_dims(np.array(X), -1), np.array(y)

    lookback_window = 30
    X_train, y_train = create_sequences(train_scaled, lookback_window)
    X_test, y_test = create_sequences(test_scaled, lookback_window)

    print(f"Sequence windows created (Lookback = {lookback_window} days):")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  y_test shape:  {y_test.shape}\n")

    # 5. Build Keras LSTM Model
    model = tf.keras.Sequential([
        layers.Input(shape=(lookback_window, 1)),
        layers.LSTM(64, activation="tanh", return_sequences=False, name="lstm_1"),
        layers.Dense(32, activation="relu", name="dense_1"),
        layers.Dense(1, name="output_layer")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )
    model.summary()

    # 6. Model Training
    epochs = 25
    batch_size = 16
    print(f"\nTraining LSTM model (Epochs: {epochs}, Batch Size: {batch_size})...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        shuffle=True,
        verbose=1
    )

    # 7. Evaluate and De-normalize predictions
    predictions_scaled = model.predict(X_test, verbose=0).flatten()
    
    # Transform predictions and targets back to Celsius scale
    predictions = descale(predictions_scaled)
    targets = descale(y_test)

    # Calculate errors
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    print(f"\nEvaluation on Unseen Test Set:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f} °C")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f} °C\n")

    # 8. Plot and Save Temperature Forecasting Dashboard
    print("Generating forecasting visualization dashboard...")
    fig, (ax_loss, ax_forecast) = plt.subplots(1, 2, figsize=(15, 6))

    # Left Plot: Loss curves
    ax_loss.plot(history.history['loss'], label='Training Loss', color='#e74c3c', linewidth=2)
    ax_loss.plot(history.history['val_loss'], label='Validation Loss', color='#3498db', linewidth=2)
    ax_loss.set_title("LSTM Model MSE Loss Curves", fontsize=11, fontweight="bold", pad=10)
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss (MSE)")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, linestyle="--", alpha=0.3)

    # Right Plot: Forecast vs Ground Truth (Last 150 days of Test Set)
    num_display_days = 150
    display_targets = targets[-num_display_days:]
    display_predictions = predictions[-num_display_days:]
    display_t = np.arange(len(display_targets))

    ax_forecast.plot(display_t, display_targets, label='True Temperature', color='#34495e', linewidth=2, zorder=2)
    ax_forecast.plot(display_t, display_predictions, label='LSTM Forecast', color='#2ecc71', linestyle='--', linewidth=2, zorder=3)
    ax_forecast.set_title(f"Temperature Forecast Comparison (Last {num_display_days} Days)", fontsize=11, fontweight="bold", pad=10)
    ax_forecast.set_xlabel("Days in Test Period")
    ax_forecast.set_ylabel("Temperature (°C)")
    
    # Print metrics directly on the plot
    metric_text = f"MAE: {mae:.2f} °C\nRMSE: {rmse:.2f} °C"
    ax_forecast.text(0.05, 0.05, metric_text, transform=ax_forecast.transAxes, fontsize=10,
                     fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.8, edgecolor='#ccc'))

    ax_forecast.legend(loc="upper right")
    ax_forecast.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_filename = "temperature_forecasting_results.png"
    plt.savefig(output_filename, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Success! Results panel saved as '{output_filename}'.")
    print("====================================================")

if __name__ == "__main__":
    main()
