import yfinance as yf
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from fastapi import FastAPI
import uvicorn
from datetime import datetime, timedelta
from send_telegram import send_telegram_message  # Убедитесь, что файл send_telegram.py существует
import optuna

app = FastAPI()

MODEL_PATH = "model.pkl"
ACCURACY_PATH = "accuracy.json"

# Параметры стратегии
HORIZON_PERIODS = 1
LOOKBACK_PERIOD = "max"
MIN_DATA_ROWS = 100
TARGET_ACCURACY = 0.8
MIN_ACCURACY_FOR_SIGNAL = 0.5
MAX_TRAINING_TIME = 3600  # В секундах

def compute_rsi(data, periods=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(data, window=20, num_std=2):
    ma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    return ma + (num_std * std), ma - (num_std * std)

def compute_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def prepare_data():
    print("Downloading data...")
    try:
        end_date = datetime.now()
        while end_date.weekday() >= 5:
            end_date -= timedelta(days=1)
        df = yf.download("EURUSD=X", interval="1d", period=LOOKBACK_PERIOD, end=end_date)
    except Exception as e:
        raise ValueError(f"Error downloading data: {str(e)}")

    if df.empty:
        raise ValueError("Empty data received from Yahoo Finance.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df[(df['Close'] > 0) & (df['Open'] > 0)]
    df['Close'] = df['Close'].fillna(method='ffill').fillna(method='bfill')
    df['Target'] = df['Close'].shift(-HORIZON_PERIODS)

    df['RSI'] = compute_rsi(df['Close'])
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_Up'], df['BB_Low'] = compute_bollinger_bands(df['Close'])
    df['Lag1'] = df['Close'].shift(1)
    df['MACD'], df['MACD_Sig'] = compute_macd(df['Close'])
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['PriceChange'] = df['Close'].pct_change()
    df = df[df['PriceChange'].abs() < 0.1]

    df.dropna(inplace=True)

    if len(df) < MIN_DATA_ROWS:
        raise ValueError(f"Insufficient data: {len(df)} rows, required {MIN_DATA_ROWS}")

    X = df[['Open', 'High', 'Low', 'Close', 'RSI', 'MA20', 'BB_Up', 'BB_Low',
            'Lag1', 'MACD', 'MACD_Sig', 'Hour', 'DayOfWeek']]
    y = (df['Target'] > df['Close']).astype(int)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    return X, y, scaler

def train_model():
    try:
        X, y, scaler = prepare_data()
    except Exception as e:
        print(f"Data preparation error: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.1),
        }

        model = LGBMClassifier(**params, random_state=42, force_col_wise=True, verbose=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)

    print("🔍 Starting Optuna hyperparameter search...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=MAX_TRAINING_TIME)

    best_params = study.best_params
    best_acc = study.best_value

    if best_acc < MIN_ACCURACY_FOR_SIGNAL:
        print(f"❌ Accuracy too low: {best_acc}")
        return

    model = LGBMClassifier(**best_params, random_state=42, force_col_wise=True, verbose=-1)
    model.fit(X_train, y_train)

    joblib.dump({'model': model, 'scaler': scaler}, MODEL_PATH)
    with open(ACCURACY_PATH, "w") as f:
        json.dump({
            "accuracy": best_acc,
            "last_trained": str(datetime.now()),
            "best_params": best_params
        }, f)

    generate_signal(model, scaler, X.iloc[-1:], X.index[-1])

def generate_signal(model, scaler, latest_data, last_index):
    try:
        scaled = scaler.transform(latest_data)
        scaled = pd.DataFrame(scaled, columns=latest_data.columns, index=latest_data.index)
        pred = model.predict(scaled)[0]
        price = latest_data['Close'].iloc[0]
        stop_loss = price * (0.99 if pred == 1 else 1.01)
        take_profit = price * (1.015 if pred == 1 else 0.985)

        signal = {
            "time": str(datetime.now()),
            "price": round(price, 5),
            "signal": "BUY" if pred == 1 else "SELL",
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5)
        }

        msg = (
            f"📊 Signal: {signal['signal']}\n"
            f"🕒 Time: {signal['time']}\n"
            f"💰 Price: {signal['price']}\n"
            f"📉 Stop Loss: {signal['stop_loss']}\n"
            f"📈 Take Profit: {signal['take_profit']}"
        )

        print(msg)
        send_telegram_message(msg)

    except Exception as e:
        print(f"Signal generation error: {e}")

@app.get("/")
async def root():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ACCURACY_PATH):
        train_model()
    else:
        with open(ACCURACY_PATH, "r") as f:
            data = json.load(f)
        last_trained = datetime.fromisoformat(data["last_trained"])
        if (datetime.now() - last_trained).days >= 1 or data["accuracy"] < TARGET_ACCURACY:
            train_model()
    return {"status": "Bot is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
