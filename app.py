import pandas as pd
import numpy as np
import requests
import joblib
import json
import os
import time
import logging

from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier, LGBMRegressor

import optuna
import optuna.samplers

from fastapi import FastAPI
import uvicorn

import config
from send_telegram import send_telegram_message

# --- Логирование ---
logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Константы ---
MODEL_PATH = config.MODEL_PATH
ACCURACY_PATH = config.ACCURACY_PATH
HORIZON_PERIODS = config.HORIZON_PERIODS
LOOKBACK_PERIOD = config.LOOKBACK_PERIOD
MIN_DATA_ROWS = config.MIN_DATA_ROWS
TARGET_ACCURACY = config.TARGET_ACCURACY
MIN_ACCURACY_FOR_SIGNAL = config.MIN_ACCURACY_FOR_SIGNAL
MAX_TRAINING_TIME = config.MAX_TRAINING_TIME
PREDICTION_PROB_THRESHOLD = config.PREDICTION_PROB_THRESHOLD
N_SPLITS_TS_CV = config.N_SPLITS_TS_CV
OPTUNA_STORAGE_URL = config.OPTUNA_STORAGE_URL
OPTUNA_STUDY_NAME = config.OPTUNA_STUDY_NAME
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "YOUR_API_KEY")

def get_twelvedata_forex_data(symbol="EUR/USD", interval="15min", outputsize=1000):
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY
    }
    response = requests.get(base_url, params=params)
    data = response.json()

    if "values" not in data:
        logger.error(f"Twelve Data API error: {data}")
        raise ValueError(f"Twelve Data error: {data.get('message', 'Unknown error')}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    df = df.astype(float)
    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }, inplace=True)
    return df

def compute_rsi(data, periods=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(data, window=20, num_std=2):
    ma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = ma + (num_std * std)
    lower = ma - (num_std * std)
    return upper, lower

def compute_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_stochastic_oscillator(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = ((close - lowest_low) / (highest_high - lowest_low + 1e-10)) * 100
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    return atr

def compute_roc(data, period=12):
    return ((data - data.shift(period)) / data.shift(period)) * 100

def prepare_data():
    logger.info("📥 Loading EUR/USD 15min data from Twelve Data...")
    df_15m = get_twelvedata_forex_data("EUR/USD", "15min", outputsize=1000)
    df_15m = df_15m[(df_15m["Close"] > 0) & (df_15m["Open"] > 0)]
    df_15m["Close"] = df_15m["Close"].ffill().bfill()

    df_features = df_15m.copy()

    # Индикаторы
    df_features["RSI"] = compute_rsi(df_features["Close"])
    df_features["MA20"] = df_features["Close"].rolling(window=20).mean()
    df_features["BB_Up"], df_features["BB_Low"] = compute_bollinger_bands(df_features["Close"])
    df_features["MACD"], df_features["MACD_Sig"] = compute_macd(df_features["Close"])
    df_features["Stoch_K"], df_features["Stoch_D"] = compute_stochastic_oscillator(df_features["High"], df_features["Low"], df_features["Close"])
    df_features["ATR"] = compute_atr(df_features["High"], df_features["Low"], df_features["Close"])
    df_features["ROC"] = compute_roc(df_features["Close"])

    # Доп признаки
    df_features["MA_96"] = df_features["Close"].rolling(window=96).mean()
    df_features["RSI_96"] = compute_rsi(df_features["Close"], periods=96)
    df_features["price_vs_ma96"] = df_features["Close"] - df_features["MA_96"]
    df_features["price_above_ma96"] = (df_features["Close"] > df_features["MA_96"]).astype(int)

    df_features["bar_in_day"] = df_features.index.to_series().diff().gt("1H").cumsum()
    daily_returns = df_features["Close"].resample("1D").last().pct_change().ffill()
    df_features["daily_return"] = daily_returns.resample("15min").ffill()

    df_features["Close_Lag1"] = df_features["Close"].shift(1)
    df_features["RSI_Lag1"] = df_features["RSI"].shift(1)
    df_features["MACD_Lag1"] = df_features["MACD"].shift(1)
    df_features["Stoch_K_Lag1"] = df_features["Stoch_K"].shift(1)
    df_features["ATR_Lag1"] = df_features["ATR"].shift(1)
    df_features["ROC_Lag1"] = compute_roc(df_features["Close_Lag1"])

    df_features["Hour"] = df_features.index.hour
    df_features["DayOfWeek"] = df_features.index.dayofweek
    df_features["DayOfMonth"] = df_features.index.day
    df_features["Month"] = df_features.index.month
    df_features["PriceChange"] = df_features["Close"].pct_change()
    df_features = df_features[df_features["PriceChange"].abs() < 0.1].dropna()

    # Будущие уровни
    df_features["FutureMax"] = df_features["Close"].rolling(window=HORIZON_PERIODS).max().shift(-HORIZON_PERIODS)
    df_features["FutureMin"] = df_features["Close"].rolling(window=HORIZON_PERIODS).min().shift(-HORIZON_PERIODS)
    df_features["Entry"] = df_features["FutureMin"]  # 👈 вход по откату

    df_features = df_features.dropna(subset=["FutureMax", "FutureMin", "Entry"])
    df_features["TakeProfit"] = df_features["FutureMax"]
    df_features["StopLoss"] = df_features["FutureMin"]

    # Целевой признак
    df_features["Target"] = ((df_features["TakeProfit"] - df_features["Entry"]) / df_features["Entry"] >= 0.005).astype(int)

    # Финальные признаки
    feature_columns = [
        "Open", "High", "Low", "Close", "RSI", "MA20", "BB_Up", "BB_Low",
        "MACD", "MACD_Sig", "Stoch_K", "Stoch_D", "ATR", "ROC", "MA_96", "RSI_96",
        "price_vs_ma96", "price_above_ma96", "daily_return", "bar_in_day",
        "Close_Lag1", "RSI_Lag1", "MACD_Lag1", "Stoch_K_Lag1", "ATR_Lag1", "ROC_Lag1",
        "Hour", "DayOfWeek", "DayOfMonth", "Month"
    ]

    X_raw = df_features[feature_columns]
    y_raw = df_features["Target"]
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_raw.columns, index=X_raw.index)

    return X_scaled_df, y_raw, scaler, df_features

# === Обучение моделей ===
def train_model():
    logger.info("🚀 Training model...")
    X, y, scaler, df = prepare_data()
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Приводим к int для корректной работы class_weight
    y_train_val_int = y_train_val.astype(int)
    y_test_int = y_test.astype(int)

    # Показываем метки классов (важно для отладки!)
    logger.info(f"Unique labels in y_train_val: {y_train_val_int.unique()}")

    def objective(trial):
        neg_count = (y_train_val_int == 0).sum()
        pos_count = (y_train_val_int == 1).sum()
        class_weight = {
            0: 1.0,
            1: neg_count / pos_count if pos_count > 0 else 1.0
        }

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "class_weight": class_weight,
        }

        model = LGBMClassifier(**params)
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_TS_CV)
        f1_scores = []
        for train_idx, val_idx in tscv.split(X_train_val):
            model.fit(X_train_val.iloc[train_idx], y_train_val_int.iloc[train_idx])
            preds = model.predict(X_train_val.iloc[val_idx])
            scores.append(f1_score(y_train_val_int.iloc[val_idx], preds, average='weighted'))

        return np.mean(scores)

    logger.info("🔍 Starting Optuna optimization...")
    study = optuna.create_study(direction="maximize", study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_STORAGE_URL, load_if_exists=True)
    study.optimize(objective, timeout=MAX_TRAINING_TIME)

    best_params = study.best_params
    best_params.update({"objective": "binary", "random_state": 42, "n_jobs": -1})
    model = LGBMClassifier(**best_params)
    model.fit(X_train_val, y_train_val)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")

    joblib.dump(model, MODEL_PATH)
    joblib.dump({"scaler": scaler, "feature_columns": list(X.columns)}, "scaler_and_features.pkl")

    with open(ACCURACY_PATH, "w") as f:
        json.dump({"f1_score": f1, "last_trained": datetime.now().isoformat()}, f)

    logger.info(f"✅ Model trained. F1 Score: {f1:.4f}")

# === Генерация сигнала ===
def generate_signal(model, scaler, latest_features, df_point):
    probas = model.predict_proba(latest_features)[0]
    buy_proba = probas[1]
    sell_proba = probas[0]

    if buy_proba >= PREDICTION_PROB_THRESHOLD:
        entry = df_point["Entry"].values[0]
        sl = df_point["StopLoss"].values[0]
        tp = df_point["TakeProfit"].values[0]

        message = f"📊 Signal: BUY\n🕒 Time: {df_point.index[0]}\n💰 Price: {entry:.5f}\n⬆️ Buy Proba: {buy_proba:.4f}\n⬇️ Sell Proba: {sell_proba:.4f}\n📉 Stop Loss: {sl:.5f}\n📈 Take Profit: {tp:.5f}"
        send_telegram_message(message)
        logger.info(f"📤 Sent signal: {message}")
    else:
        logger.info(f"❌ No confident signal. Buy proba: {buy_proba:.4f}")

# === FastAPI ===
@app.get("/")
async def root():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ACCURACY_PATH):
            logger.info("🚀 Model not found — training...")
            train_model()

        with open(ACCURACY_PATH, "r") as f:
            data = json.load(f)

        last_trained = datetime.fromisoformat(data["last_trained"])
        if (datetime.now() - last_trained).days >= 1 or data["f1_score"] < TARGET_ACCURACY:
            logger.info("🔁 Retraining model...")
            train_model()

        model = joblib.load(MODEL_PATH)
        scaler_data = joblib.load("scaler_and_features.pkl")
        scaler = scaler_data["scaler"]
        features = scaler_data["feature_columns"]

        df = get_twelvedata_forex_data("EUR/USD", "15min", 500)
        df = df.dropna()
        df["RSI"] = compute_rsi(df["Close"])
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["BB_Up"], df["BB_Low"] = compute_bollinger_bands(df["Close"])
        df["MACD"], df["MACD_Sig"] = compute_macd(df["Close"])
        df["Stoch_K"], df["Stoch_D"] = compute_stochastic_oscillator(df["High"], df["Low"], df["Close"])
        df["ATR"] = compute_atr(df["High"], df["Low"], df["Close"])
        df["ROC"] = compute_roc(df["Close"])
        df["MA_96"] = df["Close"].rolling(window=96).mean()
        df["RSI_96"] = compute_rsi(df["Close"], periods=96)
        df["price_vs_ma96"] = df["Close"] - df["MA_96"]
        df["price_above_ma96"] = (df["Close"] > df["MA_96"]).astype(int)
        df["bar_in_day"] = df.index.to_series().diff().gt("1H").cumsum()
        df["daily_return"] = df["Close"].resample("1D").last().pct_change().ffill().resample("15min").ffill()
        df["Close_Lag1"] = df["Close"].shift(1)
        df["RSI_Lag1"] = df["RSI"].shift(1)
        df["MACD_Lag1"] = df["MACD"].shift(1)
        df["Stoch_K_Lag1"] = df["Stoch_K"].shift(1)
        df["ATR_Lag1"] = df["ATR"].shift(1)
        df["ROC_Lag1"] = compute_roc(df["Close_Lag1"])
        df["Hour"] = df.index.hour
        df["DayOfWeek"] = df.index.dayofweek
        df["DayOfMonth"] = df.index.day
        df["Month"] = df.index.month
        df["PriceChange"] = df["Close"].pct_change()
        df = df[df["PriceChange"].abs() < 0.1].dropna()

        if len(df) >= 1:
            latest_features = df[features].iloc[[-1]]
            latest_original_point = df.iloc[[-1]]
            generate_signal(model, scaler, latest_features, latest_original_point)
    except Exception as e:
        logger.error(f"❌ root() error: {e}")

    return {"status": "Bot is running"}

if __name__ == "__main__":
    uvicorn.run(app, host=config.UVICORN_HOST, port=config.UVICORN_PORT)
