# Project 46: Temperature Forecasting with LSTM

This project implements a daily temperature forecasting model utilizing a Long Short-Term Memory (LSTM) recurrent network in TensorFlow 2. The pipeline synthesizes a multi-year weather dataset, structures sequences using a sliding lookback window, trains the network to predict future values, and evaluates the de-normalized predictions.

---

## Conceptual & Mathematical Overview

### 1. Long Short-Term Memory (LSTM) Architecture
LSTMs are designed to process sequential data and overcome the vanishing gradient problem of standard recurrent neural networks (RNNs) by introducing a cell state ($c_t$) and three gating mechanisms:

1. **Forget Gate ($f_t$)**: Decides what information to discard from the cell state.
   $$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
2. **Input Gate ($i_t$)**: Decides which new values to write into the cell state.
   $$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
3. **Candidate State ($\tilde{c}_t$)**: Computes new candidate values for the cell state.
   $$\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
4. **Cell State Update ($c_t$)**: Updates the memory by combining the gated previous state and current candidates.
   $$c_t = f_t \circ c_{t-1} + i_t \circ \tilde{c}_t$$
5. **Output Gate ($o_t$)**: Decides what the next hidden state should be.
   $$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$
6. **Hidden State ($h_t$)**: Computes the final recurrent state passed to the next step.
   $$h_t = o_t \circ \tanh(c_t)$$

Where:
- $x_t \in \mathbb{R}^d$ is the input vector at time step $t$.
- $h_{t-1} \in \mathbb{R}^h$ is the hidden state of the previous time step.
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation.
- $\circ$ denotes element-wise (Hadamard) multiplication.

### 2. Time-Series Preprocessing
To train LSTMs effectively, data must be scaled and windowed:
- **MinMax Normalization**: Scales the temperature $x$ to $x_{\text{norm}} \in [0, 1]$:
  $$x_{\text{norm}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$$
- **Lookback Windowing**: For a lookback window of size $W$, the model maps the sequence $\{x_{t-W}, \dots, x_{t-1}\}$ to target $y = x_t$.
- **De-normalization**: Restores predicted outputs back to degrees Celsius:
  $$x = x_{\text{norm}} \cdot (x_{\text{max}} - x_{\text{min}}) + x_{\text{min}}$$

---

## Dataset Properties

The synthetic daily temperature series ($1095$ days / 3 years) is defined by three components:
1. **Seasonality**: Annual sine wave oscillating between winter ($6^\circ\text{C}$) and summer ($30^\circ\text{C}$).
2. **Trend**: A steady warming trend ($0.002^\circ\text{C}$ per day).
3. **Noise**: Random Gaussian noise ($\sigma = 2.0^\circ\text{C}$) representing weather variance.

---

## File Structure

- **[main.py](file:///c:/Users/Dart%20Technologies/OneDrive/Desktop/tensorflow_100_projects/46_temperature_forecasting_lstm/main.py)**: Python script:
  - Generates the daily synthetic temperature dataset.
  - Applies MinMax scaling and sequence window partitioning.
  - Builds, trains (Adam, MSE loss), and validates the Keras LSTM model.
  - Calculates Celsius forecast error metrics (MAE & RMSE).
  - Plots loss curves and de-normalized test predictions `temperature_forecasting_results.png`.
- **[README.md](file:///c:/Users/Dart%20Technologies/OneDrive/Desktop/tensorflow_100_projects/46_temperature_forecasting_lstm/README.md)**: Conceptual guide and mathematical details.
- **[.gitignore](file:///c:/Users/Dart%20Technologies/OneDrive/Desktop/tensorflow_100_projects/46_temperature_forecasting_lstm/.gitignore)**: standard gitignore configurations.

---

## Installation & Running

Ensure the environment has TensorFlow 2.x and Matplotlib:
```powershell
pip install tensorflow matplotlib numpy
```

Run the pipeline:
```powershell
python main.py
```
