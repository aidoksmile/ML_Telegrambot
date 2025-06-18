import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from fastapi import FastAPI
import uvicorn
from datetime import datetime, timedelta
from send_telegram import send_telegram_message
import time
import optuna
import logging
import optuna.samplers
import config
import requests

# --- Логирование ---
logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI()

# --- Константы из config.py ---
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
MIN_ATR_SL_MULTIPLIER = config.MIN_ATR_SL_MULTIPLIER
RISK_REWARD_RATIO = config.RISK_REWARD_RATIO
BB_BUFFER_FACTOR = config.BB_BUFFER_FACTOR
MAX_REASONABLE_ATR = config.MAX_REASONABLE_ATR
MAX_TP_ATR_MULTIPLIER = config.MAX_TP_ATR_MULTIPLIER

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "YOUR_API_KEY")

# --- Twelve Data API: EUR/USD 15min ---
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

# --- Индикаторы ---
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
    try:
        df_15m = get_twelvedata_forex_data(symbol="EUR/USD", interval="15min", outputsize=1000)
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        raise

    df_15m = df_15m[(df_15m["Close"] > 0) & (df_15m["Open"] > 0)]
    df_15m["Close"] = df_15m["Close"].ffill().bfill()
    df_15m["Target"] = df_15m["Close"].shift(-HORIZON_PERIODS)

    df_features = df_15m.copy()

    # --- Технические индикаторы ---
    df_features["RSI"] = compute_rsi(df_features["Close"])
    df_features["MA20"] = df_features["Close"].rolling(window=20).mean()
    df_features["BB_Up"], df_features["BB_Low"] = compute_bollinger_bands(df_features["Close"])
    df_features["MACD"], df_features["MACD_Sig"] = compute_macd(df_features["Close"])
    df_features["Stoch_K"], df_features["Stoch_D"] = compute_stochastic_oscillator(
        df_features["High"], df_features["Low"], df_features["Close"]
    )
    df_features["ATR"] = compute_atr(df_features["High"], df_features["Low"], df_features["Close"])
    df_features["ROC"] = compute_roc(df_features["Close"])
    # --- Расширенные признаки для дневного тренда на основе 15m ---
    df_features["MA_96"] = df_features["Close"].rolling(window=96).mean()
    df_features["RSI_96"] = compute_rsi(df_features["Close"], periods=96)
    df_features["price_vs_ma96"] = df_features["Close"] - df_features["MA_96"]
    df_features["price_above_ma96"] = (df_features["Close"] > df_features["MA_96"]).astype(int)

    # --- Порядковый бар в дне ---
    df_features["bar_in_day"] = df_features.index.to_series().diff().gt("1H").cumsum()

    # --- Дневной процент изменения (через 1D resample) ---
    daily_returns = df_features["Close"].resample("1D").last().pct_change().ffill()
    df_features["daily_return"] = daily_returns.resample("15min").ffill()

    # --- Лаги и временные признаки ---
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

    df_features = df_features[df_features["PriceChange"].abs() < 0.1]
    df_features = df_features.dropna()

    if len(df_features) < MIN_DATA_ROWS:
        raise ValueError(f"Недостаточно данных для обучения: {len(df_features)}")

    df_features = df_features.dropna(subset=["Target"])
    # --- Обновлённый список признаков ---
    feature_columns = [
        "Open", "High", "Low", "Close",
        "RSI", "MA20", "BB_Up", "BB_Low",
        "MACD", "MACD_Sig", "Stoch_K", "Stoch_D", "ATR", "ROC",

        # Новые признаки дневного тренда:
        "MA_96", "RSI_96", "price_vs_ma96", "price_above_ma96",
        "daily_return", "bar_in_day",

        # Лаги
        "Close_Lag1", "RSI_Lag1", "MACD_Lag1", "Stoch_K_Lag1", "ATR_Lag1", "ROC_Lag1",

        # Временные признаки
        "Hour", "DayOfWeek", "DayOfMonth", "Month"
    ]

    X_raw = df_features[feature_columns]
    y_raw = (df_features["Target"] > df_features["Close"]).astype(int)

    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_raw.columns, index=X_raw.index)

    return X_scaled_df, y_raw, scaler, df_features


def lgbm_f1_score(y_pred, y_true):
    y_true_binary = y_true.astype(int)
    y_pred_binary = (y_pred > 0.5).astype(int)
    return 'f1_score', f1_score(y_true_binary, y_pred_binary, average='weighted'), True

def train_model():
    try:
        X, y, scaler, df_original = prepare_data()
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        return

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    neg_count = y_train_val.value_counts().get(0, 1)
    pos_count = y_train_val.value_counts().get(1, 1)
    class_weight = {0: 1.0, 1: neg_count / pos_count if pos_count > 0 else 1.0}

    def objective(trial):
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
            "class_weight": class_weight
        }

        model = LGBMClassifier(**params)
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_TS_CV)
        f1_scores = []

        for train_idx, val_idx in tscv.split(X_train_val):
            X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
            y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            f1_scores.append(f1_score(y_val, preds, average='weighted'))

        return np.mean(f1_scores)

    logger.info("🔍 Starting Optuna optimization...")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(),
        sampler=optuna.samplers.TPESampler(),
        study_name=OPTUNA_STUDY_NAME,
        storage=OPTUNA_STORAGE_URL,
        load_if_exists=True
    )
    study.optimize(objective, timeout=MAX_TRAINING_TIME)

    best_params = study.best_params
    final_model = LGBMClassifier(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_train_val, y_train_val)

    y_pred_test = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    logger.info(f"✅ Accuracy: {acc:.4f}, F1: {f1:.4f}")

    joblib.dump(final_model, MODEL_PATH)
    joblib.dump({'scaler': scaler, 'feature_columns': X.columns.tolist()}, 'scaler_and_features.pkl')
    with open(ACCURACY_PATH, "w") as f:
        json.dump({
            "accuracy": float(acc),
            "f1_score": float(f1),
            "last_trained": str(datetime.now()),
            "best_params": best_params
        }, f)

    # Сигнал после обучения
    if len(X) > 0:
        latest_features_raw = X.iloc[[-1]]
        latest_original_data_point = df_original.iloc[[-1]]
        generate_signal(final_model, scaler, latest_features_raw, latest_original_data_point)
def generate_signal(model, scaler, latest_features_raw, latest_original_data_point):
    try:
        if latest_features_raw.empty or latest_original_data_point.empty:
            logger.warning("⚠️ Missing data for signal.")
            return

        # Восстановим порядок признаков
        scaler_data = joblib.load("scaler_and_features.pkl")
        saved_columns = scaler_data["feature_columns"]
        current_features = latest_features_raw.reindex(columns=saved_columns, fill_value=np.nan)
        current_features = current_features.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        latest_scaled = scaler.transform(current_features)
        prediction_proba = model.predict_proba(latest_scaled)[0]
        current_price = latest_original_data_point["Close"].iloc[0]

        current_atr = latest_features_raw["ATR"].iloc[0]
        bb_up = latest_features_raw["BB_Up"].iloc[0]
        bb_low = latest_features_raw["BB_Low"].iloc[0]

        if current_atr > MAX_REASONABLE_ATR:
            logger.warning(f"🔺 Capping ATR from {current_atr:.5f} to {MAX_REASONABLE_ATR:.5f}")
            current_atr = MAX_REASONABLE_ATR

        buy_proba = prediction_proba[1]
        sell_proba = prediction_proba[0]
        signal_type = "HOLD"
        stop_loss = None
        take_profit = None

        if buy_proba >= PREDICTION_PROB_THRESHOLD:
            signal_type = "BUY"
            structural_sl = bb_low - current_price * BB_BUFFER_FACTOR
            atr_sl = current_price - MIN_ATR_SL_MULTIPLIER * current_atr
            stop_loss = min(structural_sl, atr_sl, current_price * 0.999)
            stop_loss = max(stop_loss, current_price - MAX_TP_ATR_MULTIPLIER * current_atr)
            risk = current_price - stop_loss
            take_profit = current_price + risk * RISK_REWARD_RATIO
            take_profit = min(take_profit, current_price + MAX_TP_ATR_MULTIPLIER * current_atr)

        elif sell_proba >= PREDICTION_PROB_THRESHOLD:
            signal_type = "SELL"
            structural_sl = bb_up + current_price * BB_BUFFER_FACTOR
            atr_sl = current_price + MIN_ATR_SL_MULTIPLIER * current_atr
            stop_loss = max(structural_sl, atr_sl, current_price * 1.001)
            stop_loss = min(stop_loss, current_price + MAX_TP_ATR_MULTIPLIER * current_atr)
            risk = stop_loss - current_price
            take_profit = current_price - risk * RISK_REWARD_RATIO
            take_profit = max(take_profit, current_price - MAX_TP_ATR_MULTIPLIER * current_atr)

        logger.info(f"📊 Signal: {signal_type}, ATR={current_atr:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}")

        signal = {
            "time": str(datetime.now()),
            "price": round(current_price, 5),
            "signal": signal_type,
            "buy_proba": round(buy_proba, 4),
            "sell_proba": round(sell_proba, 4),
            "stop_loss": round(stop_loss, 5) if stop_loss else "N/A",
            "take_profit": round(take_profit, 5) if take_profit else "N/A",
        }

        msg = (
            f"📊 Signal: {signal['signal']}\n"
            f"🕒 Time: {signal['time']}\n"
            f"💰 Price: {signal['price']}\n"
            f"⬆️ Buy Proba: {signal['buy_proba']}\n"
            f"⬇️ Sell Proba: {signal['sell_proba']}\n"
            f"📉 Stop Loss: {signal['stop_loss']}\n"
            f"📈 Take Profit: {signal['take_profit']}"
        )

        logger.info("📤 Sending signal to Telegram.")
        send_telegram_message(msg)

    except Exception as e:
        logger.error(f"❌ Signal generation error: {e}")


@app.get("/")
async def root():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ACCURACY_PATH):
            logger.info("🚀 Model not found — training...")
            train_model()
        else:
            with open(ACCURACY_PATH, "r") as f:
                data = json.load(f)
            last_trained = datetime.fromisoformat(data["last_trained"])
            metric = data.get("f1_score", 0.0)

            if (datetime.now() - last_trained).days >= 1 or metric < TARGET_ACCURACY:
                logger.info("🔁 Retraining model due to outdated metrics...")
                train_model()
            else:
                logger.info(f"✅ Model OK — last trained {last_trained}, F1 = {metric:.4f}")

                # Загрузка модели и скейлера
                model = joblib.load(MODEL_PATH)
                scaler_data = joblib.load("scaler_and_features.pkl")
                scaler = scaler_data["scaler"]
                features = scaler_data["feature_columns"]

                # Загрузка данных и признаков
                df = get_twelvedata_forex_data(symbol="EUR/USD", interval="15min", outputsize=500)
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
                daily_returns = df["Close"].resample("1D").last().pct_change().ffill()
                df["daily_return"] = daily_returns.resample("15min").ffill()

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
                else:
                    logger.warning("⚠️ Not enough recent data to generate signal.")
    except Exception as e:
        logger.error(f"❌ root() error: {e}")

    return {"status": "Bot is running"}

if __name__ == "__main__":
    uvicorn.run(app, host=config.UVICORN_HOST, port=config.UVICORN_PORT)
