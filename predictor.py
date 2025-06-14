import joblib
import os
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import ta

MODEL_FILE = "model.pkl"

def fetch_data(ticker="XAUUSD=X", period="60d", interval="15m"):
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["sma"] = ta.trend.SMAIndicator(df["Close"]).sma_indicator()
    df["target"] = (df["Close"].shift(-16) > df["Close"]).astype(int)  # Прогноз на 4 часа
    df.dropna(inplace=True)
    return df

def train_model():
    df = fetch_data()
    X = df[["Close", "rsi", "sma"]]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[INFO] Model trained. Accuracy: {acc:.2f}")
    joblib.dump(model, MODEL_FILE)
    return model

def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        print("[INFO] Loading existing model...")
        return joblib.load(MODEL_FILE)
    else:
        print("[INFO] No model found, training...")
        return train_model()

def generate_signals():
    model = load_or_train_model()
    df = fetch_data()
    latest = df.iloc[-1:]
    features = latest[["Close", "rsi", "sma"]]
    prediction = model.predict(features)[0]
    current_price = latest["Close"].values[0]

    signal = {
        "pair": "XAUUSD",
        "action": "BUY" if prediction == 1 else "SELL",
        "entry": round(current_price, 2),
        "stop_loss": round(current_price * (0.99 if prediction == 1 else 1.01), 2),
        "take_profit": round(current_price * (1.02 if prediction == 1 else 0.98), 2),
        "current_price": round(current_price, 2)
    }
    return signal
