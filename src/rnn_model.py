from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# External deps (declared in requirements.txt)
from sklearn.preprocessing import MinMaxScaler


@dataclass
class ForecastResult:
    ticker: str
    dates: List[str]
    actual_close: List[float]
    pred_close: List[float]

    current_close: float
    pred_next_close: float
    next_date: str

    rmse: float
    n_rows: int
    train_size: int
    test_size: int
    lookback: int
    epochs: int


def _next_business_day(d: pd.Timestamp) -> pd.Timestamp:
    # Simple next-business-day heuristic (skips Sat/Sun).
    nd = d + pd.Timedelta(days=1)
    while nd.weekday() >= 5:
        nd += pd.Timedelta(days=1)
    return nd


def fetch_close_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch adjusted close/close prices from Yahoo Finance via yfinance.
    Returns DataFrame with Date index and a 'Close' column.
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or len(df) == 0:
        raise ValueError("No data returned. Check ticker symbol and date range.")

    # Prefer 'Close' (always present); if multi-index columns, flatten.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if "Close" not in df.columns:
        raise ValueError("Downloaded data does not include 'Close' column.")

    out = df[["Close"]].copy()
    out = out.dropna()
    out.index = pd.to_datetime(out.index)
    return out


def make_supervised(series_scaled: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert scaled series of shape (N, 1) into supervised learning arrays:
      X: (N-lookback, lookback, 1)
      y: (N-lookback, 1)
    """
    X, y = [], []
    for i in range(lookback, len(series_scaled)):
        X.append(series_scaled[i - lookback : i, 0])
        y.append(series_scaled[i, 0])
    X = np.array(X, dtype=np.float32).reshape(-1, lookback, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y


def build_rnn_model(lookback: int):
    """
    SimpleRNN-based regressor (lightweight and close to classic 'RNN stock prediction' demos).
    """
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            SimpleRNN(64, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            SimpleRNN(64, return_sequences=False),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model


# Very small in-process cache so repeated clicks don't retrain from scratch.
_CACHE = {}


def train_and_forecast(
    ticker: str,
    start: str,
    end: str,
    lookback: int = 60,
    epochs: int = 10,
    batch_size: int = 32,
    test_ratio: float = 0.2,
) -> ForecastResult:
    """
    Train an RNN on historical close prices and return:
      - Predicted close series on the test region (aligned to dates)
      - Next-day close prediction from the last available lookback window
      - RMSE on the test region (in original price units)
    """
    key = (ticker.upper(), str(start), str(end), int(lookback), int(epochs), int(batch_size), float(test_ratio))
    if key in _CACHE:
        return _CACHE[key]

    df = fetch_close_prices(ticker, start, end)
    n_rows = len(df)
    if n_rows < lookback + 30:
        raise ValueError(f"Not enough rows ({n_rows}) for lookback={lookback}. Pick a wider date range.")

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    close = df["Close"].values.reshape(-1, 1)
    close_scaled = scaler.fit_transform(close)

    # Supervised arrays
    X_all, y_all = make_supervised(close_scaled, lookback)
    dates_all = df.index[lookback:]  # aligns with y_all

    # Train/test split
    n = len(X_all)
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_test

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_test, y_test = X_all[n_train:], y_all[n_train:]
    dates_test = dates_all[n_train:]

    model = build_rnn_model(lookback)
    # Early stopping for nicer runtime
    import tensorflow as tf
    cb = [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)]
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=cb)

    # Predict test region
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).reshape(-1)
    y_true = scaler.inverse_transform(y_test).reshape(-1)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Next-day prediction from last window
    last_window = close_scaled[-lookback:].reshape(1, lookback, 1)
    pred_next_scaled = model.predict(last_window, verbose=0)
    pred_next = float(scaler.inverse_transform(pred_next_scaled)[0, 0])

    current_close = float(close[-1, 0])
    last_date = df.index[-1]
    next_date = _next_business_day(pd.Timestamp(last_date)).strftime("%Y-%m-%d")

    result = ForecastResult(
        ticker=ticker.upper(),
        dates=[d.strftime("%Y-%m-%d") for d in dates_test],
        actual_close=[float(x) for x in y_true],
        pred_close=[float(x) for x in y_pred],
        current_close=current_close,
        pred_next_close=pred_next,
        next_date=next_date,
        rmse=rmse,
        n_rows=n_rows,
        train_size=int(n_train),
        test_size=int(n_test),
        lookback=int(lookback),
        epochs=int(epochs),
    )
    _CACHE[key] = result
    return result
