import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def fetch_data():
    df = yf.download("EURUSD=X", period="30d", interval="15m")

    if df.empty or "Close" not in df.columns:
        raise ValueError("No data fetched for EURUSD=X or 'Close' column missing.")

    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df = df.dropna()
    return df

def train_model():
    df = fetch_data()

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    features = ["Close", "rsi"]
    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, df

def generate_signals():
    model, df = train_model()

    latest_data = df[["Close", "rsi"]].iloc[-1:]
    prediction = model.predict(latest_data)[0]

    if prediction == 1:
        return {"signal": "BUY", "price": df["Close"].iloc[-1]}
    else:
        return {"signal": "SELL", "price": df["Close"].iloc[-1]}
