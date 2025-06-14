# app.py
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import json
from fastapi import FastAPI
import uvicorn
from datetime import datetime, timedelta
import numpy as np
from send_telegram import send_telegram_message

app = FastAPI()

MODEL_PATH = "model.pkl"
ACCURACY_PATH = "accuracy.json"

def prepare_data():
    # Загрузка данных
    df = yf.download("EURUSD=X", interval="15m", period="6mo")
    df['target'] = df['Close'].shift(-int(96 * 4))  # ~4 дня вперёд (15 мин свечи)
    df.dropna(inplace=True)
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = (df['target'] > df['Close']).astype(int)  # 1 - рост, 0 - падение
    return X, y

def train_model():
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.2f}")
    
    joblib.dump(model, MODEL_PATH)
    
    with open(ACCURACY_PATH, "w") as f:
        json.dump({"accuracy": acc, "last_trained": str(datetime.now())}, f)

    if acc < 0.8:
        print("Точность упала ниже 80%, нужно перетренировать.")
    else:
        generate_signal(model, X.iloc[-1:], df.iloc[-1])

def generate_signal(model, latest_data, last_row):
    prediction = model.predict(latest_data)[0]
    current_price = last_row['Close']
    stop_loss = current_price * (0.99 if prediction == 1 else 1.01)
    take_profit = current_price * (1.02 if prediction == 1 else 0.98)

    signal = {
        "time": str(datetime.now()),
        "price": round(current_price, 5),
        "signal": "BUY" if prediction == 1 else "SELL",
        "stop_loss": round(stop_loss, 5),
        "take_profit": round(take_profit, 5),
    }

    msg = (
        f"📊 Signal: {signal['signal']}\n"
        f"🕒 Time: {signal['time']}\n"
        f"💰 Price: {signal['price']}\n"
        f"📉 Stop Loss: {signal['stop_loss']}\n"
        f"📈 Take Profit: {signal['take_profit']}"
    )
    send_telegram_message(msg)

@app.get("/")
async def root():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ACCURACY_PATH):
        train_model()
    else:
        with open(ACCURACY_PATH, "r") as f:
            data = json.load(f)
        last_trained = datetime.fromisoformat(data["last_trained"])
        if (datetime.now() - last_trained).days >= 1 or data["accuracy"] < 0.8:
            train_model()
    return {"status": "Bot is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
