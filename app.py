import yfinance as yf
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from fastapi import FastAPI
import uvicorn
from datetime import datetime, timedelta
import time
import optuna
from optuna.pruners import SuccessiveHalvingPruner
import shap
import ta  # Библиотека технических индикаторов

# === Telegram ===
def send_telegram_message(msg):
    print(f"[Telegram] {msg}")

app = FastAPI()

MODEL_PATH = "model.pkl"
ACCURACY_PATH = "accuracy.json"

# Параметры стратегии
HORIZON_PERIODS = 1
LOOKBACK_PERIOD = "max"
MIN_DATA_ROWS = 100
TARGET_ACCURACY = 0.8
MIN_ACCURACY_FOR_SIGNAL = 0.5
MAX_TRAINING_TIME = 3600

def compute_features(df):
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_Up'], df['BB_Low'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband(), ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    df['MACD'], df['MACD_Sig'] = ta.trend.MACD(df['Close']).macd(), ta.trend.MACD(df['Close']).macd_signal()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['Lag1'] = df['Close'].shift(1)
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['PriceChange'] = df['Close'].pct_change()
    df = df[df['PriceChange'].abs() < 0.1]
    return df.dropna()

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
    df['Target'] = df['Close'].shift(-HORIZON_PERIODS)

    df = compute_features(df)

    X = df[['Open', 'High', 'Low', 'Close', 'RSI', 'MA20', 'BB_Up', 'BB_Low',
            'Lag1', 'MACD', 'MACD_Sig', 'ATR', 'Hour', 'DayOfWeek']]
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
        print(f"Data prep error: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for train_idx, val_idx in tscv.split(X_train):
            x_tr, x_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = LGBMClassifier(**params, random_state=42, force_col_wise=True, verbose=-1)
            model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], early_stopping_rounds=10, verbose=False)
            preds = model.predict(x_val)
            acc = accuracy_score(y_val, preds)
            scores.append(acc)

        return np.mean(scores)

    pruner = SuccessiveHalvingPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_jobs=-1, timeout=MAX_TRAINING_TIME)

    best_params = study.best_params
    best_acc = study.best_value
    print(f"✅ Best accuracy: {best_acc:.4f}, Params: {best_params}")

    if best_acc < MIN_ACCURACY_FOR_SIGNAL:
        print(f"❌ Accuracy {best_acc:.2f} too low. No model saved.")
        return

    model = LGBMClassifier(**best_params, random_state=42, force_col_wise=True, verbose=-1)
    model.fit(X_train, y_train)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)

    joblib.dump({'model': model, 'scaler': scaler}, MODEL_PATH)
    with open(ACCURACY_PATH, "w") as f:
        json.dump({
            "accuracy": best_acc,
            "last_trained": str(datetime.now()),
            "best_params": best_params
        }, f)

    generate_signal(model, scaler, X.iloc[-1:], X.index[-1])

def generate_signal(model, scaler, latest_data, last_index):
    latest_data_scaled = scaler.transform(latest_data)
    latest_data_scaled = pd.DataFrame(latest_data_scaled, columns=latest_data.columns, index=latest_data.index)

    prob = model.predict_proba(latest_data_scaled)[0][1]
    prediction = 1 if prob > 0.7 else (0 if prob < 0.3 else None)

    if prediction is None:
        print("⚠️ Low confidence signal. Skipping trade.")
        return

    current_price = latest_data['Close'].iloc[0]
    atr_value = latest_data['ATR'].iloc[0]

    stop_loss = current_price - atr_value if prediction == 1 else current_price + atr_value
    take_profit = current_price + 1.5 * atr_value if prediction == 1 else current_price - 1.5 * atr_value

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
    print(f"Signal generated:\n{msg}")
    send_telegram_message(msg)

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
