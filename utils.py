import pandas as pd
import ta

def add_indicators(df):
    df['sma'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    df['macd'] = ta.trend.macd_diff(df['Close'])
    return df.dropna()

def get_target(df):
    df["target"] = (df["Close"].shift(-12) > df["Close"]).astype(int)
    return df["target"]

def calculate_sl_tp(price, direction):
    atr = 5  # упрощённо
    if direction == 1:
        return price - atr, price + 2 * atr
    else:
        return price + atr, price - 2 * atr