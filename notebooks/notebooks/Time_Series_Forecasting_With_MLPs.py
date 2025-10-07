"""
Univariate Time Series Forecasting with MLP (multi-step)

This pipeline provides a generalised framework for training and evaluating 
a feedforward neural network (MLP) on univariate time series data.

Key features:
- Accepts any CSV with a datetime index and single target column.
- Reformats data into supervised learning format with lag inputs.
- Trains a feedforward MLP in PyTorch.
- Supports multi-step prediction (predict k terms ahead).
- Provides train/test loss curves and forecast visualisation.
"""

# Imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Utility Functions
def train_test_split(data: pd.DataFrame, n_test: int):
    """Split a univariate dataset into train and test sets."""
    return data[:-n_test], data[-n_test:]


def series_to_supervised(data, n_in: int, n_out: int = 1):
    """
    Convert a univariate series into supervised learning format.

    Args:
        data: array-like, raw time series values.
        n_in: number of lag observations (input window).
        n_out: number of steps to forecast.

    Returns:
        numpy array of shape [samples, n_in + n_out]
    """
    df = pd.DataFrame(data)
    cols = []

    # Input sequence (lags)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))

    # Forecast sequence
    for i in range(0, n_out):
        cols.append(df.shift(-i))

    # Combine and drop NaNs
    agg = pd.concat(cols, axis=1)
    agg.dropna(inplace=True)

    return agg.values


# Model Definition
class TimeSeriesMLPModel(nn.Module):
    """Feedforward MLP for univariate time series forecasting."""

    def __init__(self, n_input: int, n_nodes: int, n_out: int) -> None:
        super().__init__()
        self.lm_linear = nn.Sequential(
            nn.Linear(in_features=n_input, out_features=n_nodes),
            nn.ReLU(),
            nn.Linear(in_features=n_nodes, out_features=n_out),
        )

    def forward(self, X):
        """Forward pass."""
        return self.lm_linear(X)


# Training Function
def train_model(train_x, train_y, n_input, n_nodes, n_out, n_epochs=500, lr=0.001):
    """
    Train the MLP model on univariate time series data.

    Args:
        train_x: torch tensor, training input data.
        train_y: torch tensor, training target data.
        n_input: number of lag observations.
        n_nodes: number of hidden layer nodes.
        n_out: number of forecast steps.
        n_epochs: number of training epochs.
        lr: learning rate.

    Returns:
        model: trained PyTorch model.
        train_losses: list of training losses per epoch.
    """
    model = TimeSeriesMLPModel(n_input, n_nodes, n_out)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return model, train_losses


# Forecast Function

def forecast_future(model, history, n_input, n_out, k):
    """
    Forecast k steps into the future using the trained model.

    Args:
        model: trained PyTorch model.
        history: list or array of the time series values.
        n_input: number of lag observations.
        n_out: number of steps the model predicts at once.
        k: total number of steps to forecast.

    Returns:
        predictions: list of k forecasted values.
    """
    predictions = []
    hist = history.copy()

    while len(predictions) < k:
        x_input = torch.tensor(hist[-n_input:], dtype=torch.float32).view(1, -1)
        yhat = model(x_input).detach().numpy().flatten()
        predictions.extend(yhat.tolist())
        hist.extend(yhat.tolist())

    return predictions[:k]


# Plotting
def plot_losses(train_losses):
    """Plot training losses over epochs."""
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.show()


def plot_forecast(df, train, test, predictions, k_future):
    """
    Plot original series with train/test split and future forecasts.

    Args:
        df: original dataframe with 'Value' column.
        train: training portion of series.
        test: testing portion of series.
        predictions: forecasted values.
        k_future: number of steps predicted.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Value"], label="Original Series", color="blue")
    plt.axvline(train.index[-1], color="black", linestyle="--", label="Train/Test Split")

    # Extend time index for forecasts
    last_date = df.index[-1]
    if isinstance(last_date, (np.datetime64, pd.Timestamp)):
        future_index = pd.date_range(start=last_date, periods=k_future + 1, freq=pd.infer_freq(df.index))[1:]
    else:
        # If index is numeric
        future_index = np.arange(len(df), len(df) + k_future)

    plt.plot(future_index, predictions, label=f"{k_future}-Step Forecast", color="red", linestyle="--")

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Forecast vs Original Series")
    plt.legend()
    plt.show()


# Example Usage
if __name__ == "__main__":
    # --- User settings ---
    csv_file = "data/Alcohol_Sales.csv"   # path to CSV
    target_col = "S4248SM144NCEN"         # column to forecast
    n_input, n_nodes, n_out = 12, 100, 3  # input window, hidden nodes, forecast horizon
    n_epochs, n_test, k_future = 300, 21, 10  # training epochs, test size, forecast steps

    # Load dataset
    df = pd.read_csv(csv_file, infer_datetime_format=True, index_col=0)
    df.rename({target_col: "Value"}, inplace=True, axis=1)

    # Split data
    train, test = train_test_split(df, n_test)

    # Transform to supervised format
    data = series_to_supervised(train, n_input, n_out)
    train_x, train_y = data[:, :-n_out], data[:, -n_out:]

    # Convert to tensors
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    # Train model
    model, train_losses = train_model(train_x, train_y, n_input, n_nodes, n_out, n_epochs)

    # Plot training loss
    plot_losses(train_losses)

    # Forecast k_future steps
    history = df["Value"].tolist()
    predictions = forecast_future(model, history, n_input, n_out, k_future)

    print(f"Forecasted {k_future} steps ahead: {predictions}")

    # Plot forecast
    plot_forecast(df, train, test, predictions, k_future)
