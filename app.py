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
logger.info("🚀 App started.")

app = FastAPI()

# --- Константы из config ---
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

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "YOUR_API_KEY")

# --- Загрузка данных с Finnhub ---
def get_finnhub_forex_data(symbol="OANDA:EUR_USD", resolution="15", days=60):
    end_time = int(time.time())
    start_time = end_time - days * 86400
    url = "https://finnhub.io/api/v1/forex/candle"
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": start_time,
        "to": end_time,
        "token": FINNHUB_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data.get("s") != "ok":
        raise ValueError(f"Finnhub API error: {data.get('s')}")
    df = pd.DataFrame({
        "Datetime": pd.to_datetime(data["t"], unit="s", utc=True),
        "Open": data["o"],
        "High": data["h"],
        "Low": data["l"],
        "Close": data["c"],
        "Volume": data["v"]
    })
    df.set_index("Datetime", inplace=True)
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
    logger.info("Downloading data from Finnhub...")

    try:
        df_15m = get_finnhub_forex_data(symbol="OANDA:EUR_USD", resolution="15", days=60)
        df_1d = get_finnhub_forex_data(symbol="OANDA:EUR_USD", resolution="D", days=730)
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        raise ValueError(f"Error downloading data: {str(e)}")

    if df_15m.empty:
        raise ValueError("15m data is empty.")
    if df_1d.empty:
        logger.warning("1d data is empty. Proceeding without daily features.")

    df_15m = df_15m[(df_15m["Close"] > 0) & (df_15m["Open"] > 0)]
    df_15m["Close"] = df_15m["Close"].ffill().bfill()
    df_15m["Target"] = df_15m["Close"].shift(-HORIZON_PERIODS)

    df_features = df_15m.copy()

    df_features["RSI"] = compute_rsi(df_features["Close"])
    df_features["MA20"] = df_features["Close"].rolling(window=20).mean()
    df_features["BB_Up"], df_features["BB_Low"] = compute_bollinger_bands(df_features["Close"])
    df_features["MACD"], df_features["MACD_Sig"] = compute_macd(df_features["Close"])
    df_features["Stoch_K"], df_features["Stoch_D"] = compute_stochastic_oscillator(
        df_features["High"], df_features["Low"], df_features["Close"]
    )
    df_features["ATR"] = compute_atr(df_features["High"], df_features["Low"], df_features["Close"])
    df_features["ROC"] = compute_roc(df_features["Close"])

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

    # Добавим дневные фичи
    if not df_1d.empty:
        df_1d = df_1d[(df_1d["Close"] > 0) & (df_1d["Open"] > 0)]
        df_1d["Close"] = df_1d["Close"].ffill().bfill()

        df_1d_features = pd.DataFrame(index=df_1d.index)
        df_1d_features["RSI_1d"] = compute_rsi(df_1d["Close"])
        df_1d_features["MA20_1d"] = df_1d["Close"].rolling(window=20).mean()
        df_1d_features["MACD_1d"], df_1d_features["MACD_Sig_1d"] = compute_macd(df_1d["Close"])
        df_1d_features["ATR_1d"] = compute_atr(df_1d["High"], df_1d["Low"], df_1d["Close"])
        df_1d_features["ROC_1d"] = compute_roc(df_1d["Close"])
        df_1d_features["Daily_Change"] = df_1d["Close"].pct_change()

        df_1d_features_resampled = df_1d_features.resample('15min').ffill()
        df_features = df_features.merge(df_1d_features_resampled, left_index=True, right_index=True, how='left')
    else:
        logger.warning("Skipping daily features merge due to empty df_1d.")

    df_features = df_features.dropna()
    if len(df_features) < MIN_DATA_ROWS:
        raise ValueError(f"Insufficient data: {len(df_features)} rows.")

    df_features = df_features.dropna(subset=["Target"])

    feature_columns = [
        "Open", "High", "Low", "Close", "RSI", "MA20", "BB_Up", "BB_Low", "MACD", "MACD_Sig",
        "Stoch_K", "Stoch_D", "ATR", "ROC",
        "Close_Lag1", "RSI_Lag1", "MACD_Lag1", "Stoch_K_Lag1", "ATR_Lag1", "ROC_Lag1",
        "Hour", "DayOfWeek", "DayOfMonth", "Month"
    ]

    # Добавим дневные признаки, если есть
    for col in ["RSI_1d", "MA20_1d", "MACD_1d", "MACD_Sig_1d", "ATR_1d", "ROC_1d", "Daily_Change"]:
        if col in df_features.columns:
            feature_columns.append(col)

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
        logger.error(f"Data preparation failed: {e}")
        return

    if len(X) != len(y):
        raise ValueError("X and y must have same length")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    neg_count = y_train_val.value_counts().get(0, 0)
    pos_count = y_train_val.value_counts().get(1, 0)
    class_weight = {0: 1.0, 1: neg_count / pos_count if pos_count else 1.0}

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
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

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=lgbm_f1_score,
                callbacks=[optuna.integration.LightGBMPruningCallback(trial, "f1_score")]
            )
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
    best_f1_score = study.best_value

    logger.info(f"✅ Best F1 score: {best_f1_score:.4f}")
    logger.info(f"📋 Best params: {best_params}")

    if best_f1_score < MIN_ACCURACY_FOR_SIGNAL:
        logger.warning("Model not accurate enough. Skipping save.")
        return

    final_model = LGBMClassifier(**best_params, random_state=42, n_jobs=-1, verbose=-1)
    final_model.fit(X_train_val, y_train_val)

    y_pred_test = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    logger.info(f"Final model accuracy: {acc:.4f}, F1: {f1:.4f}")

    joblib.dump(final_model, MODEL_PATH)
    joblib.dump({'scaler': scaler, 'feature_columns': X.columns.tolist()}, 'scaler_and_features.pkl')

    with open(ACCURACY_PATH, "w") as f:
        json.dump({
            "accuracy": float(acc),
            "f1_score": float(f1),
            "last_trained": str(datetime.now()),
            "best_params": best_params
        }, f)

    # Сигнал на последней точке
    if len(X) > 0:
        latest_features_raw = X.iloc[[-1]]
        latest_original_data_point = df_original.iloc[[-1]]
        generate_signal(final_model, scaler, latest_features_raw, latest_original_data_point)
@app.get("/")
async def root():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ACCURACY_PATH):
        logger.info("Model or metrics not found — training new model.")
        train_model()
    else:
        with open(ACCURACY_PATH, "r") as f:
            data = json.load(f)
        last_trained = datetime.fromisoformat(data["last_trained"])
        current_metric = data.get("f1_score", data.get("accuracy", 0.0))

        if (datetime.now() - last_trained).days >= 1 or current_metric < TARGET_ACCURACY:
            logger.info(f"Retraining required: last trained {last_trained}, metric: {current_metric}")
            train_model()
        else:
            logger.info(f"Model is up to date. Last trained: {last_trained}, metric: {current_metric:.2f}")
            try:
                model = joblib.load(MODEL_PATH)
                scaler_data = joblib.load("scaler_and_features.pkl")
                scaler = scaler_data["scaler"]
                saved_feature_columns = scaler_data["feature_columns"]

                # Загрузим последние 7 дней 15m и 30 дней 1d
                df_latest_15m = get_finnhub_forex_data("OANDA:EUR_USD", "15", days=7)
                df_latest_1d = get_finnhub_forex_data("OANDA:EUR_USD", "D", days=30)

                df_latest_15m['RSI'] = compute_rsi(df_latest_15m["Close"])
                df_latest_15m['MA20'] = df_latest_15m["Close"].rolling(window=20).mean()
                df_latest_15m['BB_Up'], df_latest_15m['BB_Low'] = compute_bollinger_bands(df_latest_15m["Close"])
                df_latest_15m['MACD'], df_latest_15m['MACD_Sig'] = compute_macd(df_latest_15m["Close"])
                df_latest_15m['Stoch_K'], df_latest_15m['Stoch_D'] = compute_stochastic_oscillator(
                    df_latest_15m["High"], df_latest_15m["Low"], df_latest_15m["Close"]
                )
                df_latest_15m['ATR'] = compute_atr(df_latest_15m["High"], df_latest_15m["Low"], df_latest_15m["Close"])
                df_latest_15m['ROC'] = compute_roc(df_latest_15m["Close"])

                df_latest_15m['Close_Lag1'] = df_latest_15m['Close'].shift(1)
                df_latest_15m['RSI_Lag1'] = df_latest_15m['RSI'].shift(1)
                df_latest_15m['MACD_Lag1'] = df_latest_15m['MACD'].shift(1)
                df_latest_15m['Stoch_K_Lag1'] = df_latest_15m['Stoch_K'].shift(1)
                df_latest_15m['ATR_Lag1'] = df_latest_15m['ATR'].shift(1)
                df_latest_15m['ROC_Lag1'] = compute_roc(df_latest_15m['Close_Lag1'])

                df_latest_15m['Hour'] = df_latest_15m.index.hour
                df_latest_15m['DayOfWeek'] = df_latest_15m.index.dayofweek
                df_latest_15m['DayOfMonth'] = df_latest_15m.index.day
                df_latest_15m['Month'] = df_latest_15m.index.month
                df_latest_15m['PriceChange'] = df_latest_15m['Close'].pct_change()
                df_latest_15m = df_latest_15m[df_latest_15m['PriceChange'].abs() < 0.1]

                if not df_latest_1d.empty:
                    df_1d_features = pd.DataFrame(index=df_latest_1d.index)
                    df_1d_features["RSI_1d"] = compute_rsi(df_latest_1d["Close"])
                    df_1d_features["MA20_1d"] = df_latest_1d["Close"].rolling(window=20).mean()
                    df_1d_features["MACD_1d"], df_1d_features["MACD_Sig_1d"] = compute_macd(df_latest_1d["Close"])
                    df_1d_features["ATR_1d"] = compute_atr(df_latest_1d["High"], df_latest_1d["Low"], df_latest_1d["Close"])
                    df_1d_features["ROC_1d"] = compute_roc(df_latest_1d["Close"])
                    df_1d_features["Daily_Change"] = df_latest_1d["Close"].pct_change()
                    df_1d_features_resampled = df_1d_features.resample("15min").ffill()
                    df_latest_15m = df_latest_15m.merge(df_1d_features_resampled, left_index=True, right_index=True, how='left')

                df_latest_15m = df_latest_15m.dropna()
                if len(df_latest_15m) >= 1:
                    latest_features_raw = df_latest_15m[saved_feature_columns].iloc[[-1]]
                    latest_original_data_point = df_latest_15m.iloc[[-1]]
                    generate_signal(model, scaler, latest_features_raw, latest_original_data_point)
                else:
                    logger.warning("Not enough fresh data to generate signal.")
            except Exception as e:
                logger.error(f"Error generating signal from latest data: {e}")

    return {"status": "Bot is running"}

if __name__ == "__main__":
    uvicorn.run(app, host=config.UVICORN_HOST, port=config.UVICORN_PORT)
