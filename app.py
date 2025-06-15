import yfinance as yf
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
from send_telegram import send_telegram_message # <-- РАСКОММЕНТИРОВАНО для реальной отправки
import time
import optuna
import logging
import optuna.samplers

import config

# --- Настройка логирования ---
logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
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
OPTUNA_STORAGE_URL = config.OPTUNA_STORAGE_URL # <-- Добавлено
OPTUNA_STUDY_NAME = config.OPTUNA_STUDY_NAME # <-- Добавлено

# УДАЛЕНА ЗАГЛУШКА send_telegram_message, теперь используется импортированная функция

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

def prepare_data():
    logger.info("Downloading data...")
    try:
        end_date = datetime.now()
        while end_date.weekday() >= 5:
            end_date -= timedelta(days=1)
        df = yf.download("EURUSD=X", interval="1d", period=LOOKBACK_PERIOD, end=end_date)
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        raise ValueError(f"Error downloading data: {str(e)}")

    if df.empty:
        logger.error("Empty data received from Yahoo Finance.")
        raise ValueError("Empty data received from Yahoo Finance.")

    logger.info(f"Downloaded {len(df)} rows, from {df.index[0]} to {df.index[-1]}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df[(df['Close'] > 0) & (df['Open'] > 0)]
    logger.info(f"After filtering Open/Close > 0: {len(df)} rows")

    df['Close'] = df['Close'].fillna(method='ffill').fillna(method='bfill')
    df['Target'] = df['Close'].shift(-HORIZON_PERIODS)

    if df['Target'].isna().all():
        logger.error("Target column contains only NaNs.")
        raise ValueError("Target column contains only NaNs.")

    # Создаем копию df для расчета признаков, чтобы не модифицировать исходный df
    df_features = df.copy() 
    df_features['RSI'] = compute_rsi(df_features['Close'])
    df_features['MA20'] = df_features['Close'].rolling(window=20).mean()
    df_features['BB_Up'], df_features['BB_Low'] = compute_bollinger_bands(df_features['Close'])
    df_features['Lag1'] = df_features['Close'].shift(1)
    df_features['MACD'], df_features['MACD_Sig'] = compute_macd(df_features['Close'])
    df_features['Hour'] = df_features.index.hour
    df_features['DayOfWeek'] = df_features.index.dayofweek
    df_features['PriceChange'] = df_features['Close'].pct_change()
    df_features = df_features[df_features['PriceChange'].abs() < 0.1] 

    initial_rows = len(df_features)
    df_features = df_features.dropna() 
    logger.info(f"After dropna: {len(df_features)} rows (dropped {initial_rows - len(df_features)})")

    if len(df_features) < MIN_DATA_ROWS:
        logger.error(f"Insufficient data: {len(df_features)} rows, required {MIN_DATA_ROWS}")
        raise ValueError(f"Insufficient data: {len(df_features)} rows, required {MIN_DATA_ROWS}")

    X = df_features[['Open', 'High', 'Low', 'Close', 'RSI', 'MA20', 'BB_Up', 'BB_Low',
            'Lag1', 'MACD', 'MACD_Sig', 'Hour', 'DayOfWeek']]
    y = (df_features['Target'] > df_features['Close']).astype(int)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    logger.info(f"X shape: {X.shape}, y distribution: {y.value_counts().to_dict()}")
    # Возвращаем также df_features, чтобы получить доступ к немасштабированным ценам
    return X, y, scaler, df_features 

def train_model():
    try:
        X, y, scaler, df_original = prepare_data() # <-- Получаем df_original
    except Exception as e:
        logger.error(f"Data preparation error, cannot train model: {e}")
        return

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    logger.info(f"Train/Validation set size: {len(X_train_val)}, Test set size: {len(X_test)}")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.1),
            'random_state': 42,
            'force_col_wise': True,
            'verbose': -1,
            'n_jobs': -1,
        }

        model = LGBMClassifier(**params)
        
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_TS_CV)
        f1_scores = []

        for train_index, val_index in tscv.split(X_train_val):
            X_fold_train, X_fold_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
            y_fold_train, y_fold_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

            model.fit(X_fold_train, y_fold_train)
            preds = model.predict(X_fold_val)
            f1_scores.append(f1_score(y_fold_val, preds))

        avg_f1 = np.mean(f1_scores)
        
        trial.report(avg_f1, trial.number)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return avg_f1

    logger.info("🔍 Starting Optuna hyperparameter search...")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(),
        sampler=optuna.samplers.TPESampler(),
        study_name=OPTUNA_STUDY_NAME,
        storage=OPTUNA_STORAGE_URL,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=None, timeout=MAX_TRAINING_TIME)

    best_params = study.best_params
    best_f1_score = study.best_value

    logger.info(f"✅ Optuna best F1-score: {best_f1_score:.4f}")
    logger.info(f"📋 Best params: {best_params}")

    if best_f1_score < MIN_ACCURACY_FOR_SIGNAL:
        logger.warning(f"❌ F1-score {best_f1_score:.2f} too low. No model saved.")
        return

    best_model = LGBMClassifier(**best_params, random_state=42, force_col_wise=True, verbose=-1)
    best_model.fit(X_train_val, y_train_val)

    test_preds = best_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, test_preds)
    final_f1_score = f1_score(y_test, test_preds)
    logger.info(f"Final model performance on TEST set: Accuracy={final_accuracy:.4f}, F1-score={final_f1_score:.4f}")

    joblib.dump({'model': best_model, 'scaler': scaler}, MODEL_PATH)
    with open(ACCURACY_PATH, "w") as f:
        json.dump({
            "accuracy": final_accuracy,
            "f1_score": final_f1_score,
            "last_trained": str(datetime.now()),
            "best_params": best_params
        }, f)

    # Генерируем сигнал на основе последней доступной точки данных (немасштабированной)
    if not df_original.empty: # <-- Используем df_original
        generate_signal(best_model, scaler, df_original.iloc[-1:], df_original.index[-1])
    else:
        logger.warning("No data to generate signal after training.")

def generate_signal(model, scaler, latest_data, last_index):
    try:
        if latest_data.empty:
            logger.warning("No latest data to generate signal.")
            return

        # Масштабируем только те признаки, которые используются моделью для предсказания
        # Убедимся, что latest_data содержит все необходимые колонки для X
        features_for_scaling = latest_data[['Open', 'High', 'Low', 'Close', 'RSI', 'MA20', 'BB_Up', 'BB_Low',
                                            'Lag1', 'MACD', 'MACD_Sig', 'Hour', 'DayOfWeek']]
        
        latest_data_scaled = scaler.transform(features_for_scaling)
        latest_data_scaled = pd.DataFrame(latest_data_scaled, columns=features_for_scaling.columns, index=latest_data.index)
        
        prediction_proba = model.predict_proba(latest_data_scaled)[0]
        
        # current_price берется из немасштабированных данных
        current_price = latest_data['Close'].iloc[0] 
        
        signal_type = "HOLD"
        stop_loss = None
        take_profit = None

        if prediction_proba[1] >= PREDICTION_PROB_THRESHOLD:
            signal_type = "BUY"
            stop_loss = current_price * 0.99
            take_profit = current_price * 1.015
        elif prediction_proba[0] >= PREDICTION_PROB_THRESHOLD:
            signal_type = "SELL"
            stop_loss = current_price * 1.01
            take_profit = current_price * 0.985
        
        signal = {
            "time": str(datetime.now()),
            "price": round(current_price, 5),
            "signal": signal_type,
            "buy_proba": round(prediction_proba[1], 4),
            "sell_proba": round(prediction_proba[0], 4),
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
        logger.info(f"Signal generated:\n{msg}")
        send_telegram_message(msg)
    except Exception as e:
        logger.error(f"Signal generation error: {e}")

@app.get("/")
async def root():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ACCURACY_PATH):
        logger.info("Model or accuracy file not found. Training new model.")
        train_model()
    else:
        with open(ACCURACY_PATH, "r") as f:
            data = json.load(f)
        last_trained = datetime.fromisoformat(data["last_trained"])
        
        current_metric = data.get("f1_score", data.get("accuracy", 0.0))
        
        if (datetime.now() - last_trained).days >= 1 or current_metric < TARGET_ACCURACY:
            logger.info(f"Model needs retraining. Last trained: {last_trained}, Metric: {current_metric:.2f}")
            train_model()
        else:
            logger.info(f"Model is up to date. Last trained: {last_trained}, Metric: {current_metric:.2f}")
            try:
                model_data = joblib.load(MODEL_PATH)
                model = model_data['model']
                scaler = model_data['scaler']
                
                end_date = datetime.now()
                while end_date.weekday() >= 5:
                    end_date -= timedelta(days=1)
                # Загружаем достаточно данных для расчета индикаторов для последней точки
                # Важно: df_latest должен быть полным DataFrame для расчета индикаторов
                df_latest_full = yf.download("EURUSD=X", interval="1d", period="30d", end=end_date) 
                if isinstance(df_latest_full.columns, pd.MultiIndex):
                    df_latest_full.columns = [col[0] for col in df_latest_full.columns]
                df_latest_full['Close'] = df_latest_full['Close'].fillna(method='ffill').fillna(method='bfill')
                
                # Пересчитываем признаки для последних данных
                df_latest_full['RSI'] = compute_rsi(df_latest_full['Close'])
                df_latest_full['MA20'] = compute_rsi(df_latest_full['Close']) # Ошибка: было compute_rsi, должно быть rolling mean
                df_latest_full['MA20'] = df_latest_full['Close'].rolling(window=20).mean() # <-- ИСПРАВЛЕНО
                df_latest_full['BB_Up'], df_latest_full['BB_Low'] = compute_bollinger_bands(df_latest_full['Close'])
                df_latest_full['Lag1'] = df_latest_full['Close'].shift(1)
                df_latest_full['MACD'], df_latest_full['MACD_Sig'] = compute_macd(df_latest_full['Close'])
                df_latest_full['Hour'] = df_latest_full.index.hour
                df_latest_full['DayOfWeek'] = df_latest_full.index.dayofweek
                df_latest_full['PriceChange'] = df_latest_full['Close'].pct_change()
                df_latest_full = df_latest_full[df_latest_full['PriceChange'].abs() < 0.1]
                df_latest_full = df_latest_full.dropna()

                if not df_latest_full.empty:
                    # Передаем полную последнюю строку с рассчитанными признаками и оригинальной ценой
                    generate_signal(model, scaler, df_latest_full.iloc[-1:], df_latest_full.index[-1])
                else:
                    logger.warning("Could not get enough latest data to generate signal from existing model.")
            except Exception as e:
                logger.error(f"Error loading model or generating signal from existing model: {e}")

    return {"status": "Bot is running"}

if __name__ == "__main__":
    uvicorn.run(app, host=config.UVICORN_HOST, port=config.UVICORN_PORT)
