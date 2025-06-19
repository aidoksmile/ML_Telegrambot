import pandas as pd
import numpy as np
import requests
import joblib
import json
import os
import time
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import optuna
from fastapi import FastAPI
import uvicorn
import config
from send_telegram import send_telegram_message

# Логирование
logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Константы
MODEL_PATH = config.MODEL_PATH
ACCURACY_PATH = config.ACCURACY_PATH
HORIZON_PERIODS = 12  # ~3 часа для 15мин
LOOKBACK_PERIOD = 96  # ~1 день
MIN_DATA_ROWS = config.MIN_DATA_ROWS
TARGET_ACCURACY = 0.75
MIN_ACCURACY_FOR_SIGNAL = config.MIN_ACCURACY_FOR_SIGNAL
MAX_TRAINING_TIME = 3600  # 1 час
PREDICTION_PROB_THRESHOLD = 0.7
N_SPLITS_TS_CV = 5
OPTUNA_STORAGE_URL = config.OPTUNA_STORAGE_URL
OPTUNA_STUDY_NAME = config.OPTUNA_STUDY_NAME
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "YOUR_API_KEY")

def get_twelvedata_forex_data(symbol="EUR/USD", interval="15min", outputsize=1500):
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

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    return atr

def prepare_data():
    logger.info("📥 Loading EUR/USD 15min data...")
    df_15m = get_twelvedata_forex_data("EUR/USD", "15min", outputsize=1500)
    df_15m = df_15m[(df_15m["Close"] > 0) & (df_15m["Open"] > 0)]
    df_15m["Close"] = df_15m["Close"].ffill().bfill()

    df_features = df_15m.copy()

    # Индикаторы
    df_features["RSI"] = compute_rsi(df_features["Close"], periods=14)
    df_features["MA20"] = df_features["Close"].rolling(window=20).mean()
    df_features["BB_Up"], df_features["BB_Low"] = compute_bollinger_bands(df_features["Close"])
    df_features["MACD"], df_features["MACD_Sig"] = compute_macd(df_features["Close"])
    df_features["ATR"] = compute_atr(df_features["High"], df_features["Low"], df_features["Close"])

    # Дополнительные признаки
    df_features["MA_96"] = df_features["Close"].rolling(window=96).mean()
    df_features["price_vs_ma96"] = df_features["Close"] - df_features["MA_96"]
    df_features["Hour"] = df_features.index.hour
    df_features["DayOfWeek"] = df_features.index.dayofweek

    # Лаги
    df_features["Close_Lag1"] = df_features["Close"].shift(1)
    df_features["RSI_Lag1"] = df_features["RSI"].shift(1)

    # Целевой признак
    df_features["FutureReturn"] = df_features["Close"].shift(-HORIZON_PERIODS) / df_features["Close"] - 1
    df_features["Target"] = (df_features["FutureReturn"] >= 0.005).astype(int)

    df_features = df_features.dropna()

    # Логирование распределения целевого признака
    logger.info(f"Target distribution:\n{df_features['Target'].value_counts(normalize=True)}")

    # Финальные признаки
    feature_columns = [
        "Open", "High", "Low", "Close", "RSI", "MA20", "BB_Up", "BB_Low",
        "MACD", "MACD_Sig", "ATR", "MA_96", "price_vs_ma96",
        "Hour", "DayOfWeek", "Close_Lag1", "RSI_Lag1"
    ]

    X_raw = df_features[feature_columns]
    y_raw = df_features["Target"]
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_raw.columns, index=X_raw.index)

    return X_scaled_df, y_raw, scaler, df_features

def train_model():
    logger.info("🚀 Training model...")
    X, y, scaler, df = prepare_data()
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS_TS_CV)
    
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 40),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42,
            "verbosity": -1
        }

        model = LGBMClassifier(**params)
        scores = []
        for train_idx, val_idx in tscv.split(X_train_val):
            X_tr, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
            y_tr, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
            if len(np.unique(y_tr)) > 1:
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                score = f1_score(y_val, preds, average="weighted")
                scores.append(score)
        return np.mean(scores) if scores else 0.0

    logger.info("🔍 Starting Optuna optimization...")
    study = optuna.create_study(direction="maximize", study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_STORAGE_URL, load_if_exists=True)
    study.optimize(objective, timeout=MAX_TRAINING_TIME)

    best_params = study.best_params
    best_params.update({"objective": "binary", "random_state": 42, "n_jobs": -1})
    model = LGBMClassifier(**best_params)
    model.fit(X_train_val, y_train_val)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    logger.info(f"✅ Model trained. F1 Score: {f1:.4f}")

    if f1 < 0.6 or f1 > 0.95:
        logger.warning("⚠️ Suspicious F1 score. Possible overfitting or data issue.")

    # Логирование важности признаков
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    logger.info(f"Feature importance:\n{importance}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump({"scaler": scaler, "feature_columns": list(X.columns)}, "scaler_and_features.pkl")

    with open(ACCURACY_PATH, "w") as f:
        json.dump({"f1_score": f1, "last_trained": datetime.now().isoformat()}, f)

def generate_signal(model, scaler, latest_features, df_point):
    probas = model.predict_proba(latest_features)[0]
    buy_proba = probas[1]
    if buy_proba >= PREDICTION_PROB_THRESHOLD:
        entry = df_point["Close"].values[0]
        message = f"📊 Signal: BUY\n🕒 Time: {df_point.index[0]}\n💰 Price: {entry:.5f}\n⬆️ Buy Proba: {buy_proba:.4f}"
        send_telegram_message(message)
        logger.info(f"📤 Sent signal: {message}")
    else:
        logger.info(f"❌ No confident signal. Buy proba: {buy_proba:.4f}")

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
        df["ATR"] = compute_atr(df["High"], df["Low"], df["Close"])
        df["MA_96"] = df["Close"].rolling(window=96).mean()
        df["price_vs_ma96"] = df["Close"] - df["MA_96"]
        df["Hour"] = df.index.hour
        df["DayOfWeek"] = df.index.dayofweek
        df["Close_Lag1"] = df["Close"].shift(1)
        df["RSI_Lag1"] = df["RSI"].shift(1)
        if len(df) >= 1:
            latest_features = df[features].iloc[[-1]]
            latest_features = scaler.transform(latest_features)
            latest_features = pd.DataFrame(latest_features, columns=features)
            latest_original_point = df.iloc[[-1]]
            generate_signal(model, scaler, latest_features, latest_original_point)
    except Exception as e:
        logger.error(f"❌ root() error: {e}")
    return {"status": "Bot is running"}

if __name__ == "__main__":
    uvicorn.run(app, host=config.UVICORN_HOST, port=config.UVICORN_PORT)
