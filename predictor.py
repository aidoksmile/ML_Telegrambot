import yfinance as yf
import pandas as pd
import joblib
import os
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier

def fetch_data():
    df = yf.download("EURUSD=X", interval="15m", period="7d")
    df.dropna(inplace=True)
    df["rsi"] = RSIIndicator(close=df["Close"]).rsi()
    df.dropna(inplace=True)
    return df

def train_model():
    df = fetch_data()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    X = df[["rsi"]]
    y = df["target"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    return model

def load_or_train_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return train_model()

def generate_signals():
    model = load_or_train_model()
    df = fetch_data()
    latest_data = df[["rsi"]].iloc[-1:]
    prediction = model.predict(latest_data)[0]
    return "BUY" if prediction == 1 else "SELL"
