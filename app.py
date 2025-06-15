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
from send_telegram import send_telegram_message
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
LOOKBACK_PERIOD = config.LOOKBACK_PERIOD # Теперь "60d"
MIN_DATA_ROWS = config.MIN_DATA_ROWS
TARGET_ACCURACY = config.TARGET_ACCURACY
MIN_ACCURACY_FOR_SIGNAL = config.MIN_ACCURACY_FOR_SIGNAL
MAX_TRAINING_TIME = config.MAX_TRAINING_TIME
PREDICTION_PROB_THRESHOLD = config.PREDICTION_PROB_THRESHOLD
N_SPLITS_TS_CV = config.N_SPLITS_TS_CV
OPTUNA_STORAGE_URL = config.OPTUNA_STORAGE_URL
OPTUNA_STUDY_NAME = config.OPTUNA_STUDY_NAME

# --- Вспомогательные функции для технических индикаторов ---
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
    """Calculates Stochastic Oscillator %K and %D."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = ((close - lowest_low) / (highest_high - lowest_low + 1e-10)) * 100
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def compute_atr(high, low, close, period=14):
    """Calculates Average True Range (ATR)."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    return atr

def compute_roc(data, period=12):
    """Calculates Rate of Change (ROC)."""
    roc = ((data - data.shift(period)) / data.shift(period)) * 100
    return roc

def prepare_data():
    logger.info("Downloading data...")
    try:
        end_date = datetime.now()
        # Для 15-минутных данных нет необходимости пропускать выходные,
        # так как рынок форекс работает 24/5.
        # Однако, Yahoo Finance может не предоставлять данные за выходные.
        df = yf.download("EURUSD=X", interval="15m", period=LOOKBACK_PERIOD, end=end_date) # <-- ИЗМЕНЕНО
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        raise ValueError(f"Error downloading data: {str(e)}")

    if df.empty:
        logger.error("Empty data received from Yahoo Finance.")
        raise ValueError("Empty data received from Yahoo Finance.")

    logger.info(f"Downloaded {len(df)} rows, from {df.index[0]} to {df.index[-1]} for 15m interval.") # <-- Обновлено лог-сообщение
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df[(df['Close'] > 0) & (df['Open'] > 0)]
    logger.info(f"After filtering Open/Close > 0: {len(df)} rows")

    # Проверка на наличие и значения Volume
    if 'Volume' in df.columns and df['Volume'].sum() == 0:
        logger.warning("Volume data is present but all values are zero. This is common for forex data from Yahoo Finance.")
    elif 'Volume' not in df.columns:
        logger.warning("Volume column not found in downloaded data. It will not be used as a feature.")
        df['Volume'] = 0 # Добавляем столбец с нулями, чтобы избежать ошибок, если он отсутствует

    df['Close'] = df['Close'].fillna(method='ffill').fillna(method='bfill')
    df['Target'] = df['Close'].shift(-HORIZON_PERIODS)

    if df['Target'].isna().all():
        logger.error("Target column contains only NaNs.")
        raise ValueError("Target column contains only NaNs.")

    # Создаем копию df для расчета признаков, чтобы не модифицировать исходный df
    df_features = df.copy() 

    # --- Добавление новых и существующих индикаторов ---
    df_features['RSI'] = compute_rsi(df_features['Close'])
    df_features['MA20'] = df_features['Close'].rolling(window=20).mean()
    df_features['BB_Up'], df_features['BB_Low'] = compute_bollinger_bands(df_features['Close'])
    df_features['MACD'], df_features['MACD_Sig'] = compute_macd(df_features['Close'])
    
    df_features['Stoch_K'], df_features['Stoch_D'] = compute_stochastic_oscillator(df_features['High'], df_features['Low'], df_features['Close'])
    df_features['ATR'] = compute_atr(df_features['High'], df_features['Low'], df_features['Close'])
    df_features['ROC'] = compute_roc(df_features['Close'])

    # --- Добавление лаговых признаков для нескольких индикаторов ---
    df_features['Close_Lag1'] = df_features['Close'].shift(1)
    df_features['RSI_Lag1'] = df_features['RSI'].shift(1)
    df_features['MACD_Lag1'] = df_features['MACD'].shift(1)
    df_features['Stoch_K_Lag1'] = df_features['Stoch_K'].shift(1)
    df_features['ATR_Lag1'] = df_features['ATR'].shift(1)
    df_features['ROC_Lag1'] = df_features['ROC'].shift(1)
    df_features['Volume_Lag1'] = df_features['Volume'].shift(1)

    # --- Временные признаки ---
    # Для 15-минутных данных DayOfMonth и Month могут быть менее значимы,
    # но Hour и DayOfWeek по-прежнему важны.
    df_features['Hour'] = df_features.index.hour
    df_features['DayOfWeek'] = df_features.index.dayofweek
    df_features['DayOfMonth'] = df_features.index.day
    df_features['Month'] = df_features.index.month

    # --- Обработка выбросов (уже есть, но убедимся) ---
    # Для 15-минутных данных порог 0.1 может быть слишком высоким,
    # так как процентные изменения обычно меньше.
    # Возможно, стоит пересмотреть этот порог или убрать его,
    # если он отфильтровывает слишком много данных.
    df_features['PriceChange'] = df_features['Close'].pct_change()
    df_features = df_features[df_features['PriceChange'].abs() < 0.1] 

    initial_rows = len(df_features)
    df_features = df_features.dropna() 
    logger.info(f"After dropna: {len(df_features)} rows (dropped {initial_rows - len(df_features)})")

    if len(df_features) < MIN_DATA_ROWS:
        logger.error(f"Insufficient data: {len(df_features)} rows, required {MIN_DATA_ROWS}")
        raise ValueError(f"Insufficient data: {len(df_features)} rows, required {MIN_DATA_ROWS}")

    # --- Определение признаков для X ---
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 
        'RSI', 'MA20', 'BB_Up', 'BB_Low', 'MACD', 'MACD_Sig',
        'Stoch_K', 'Stoch_D', 'ATR', 'ROC',
        'Close_Lag1', 'RSI_Lag1', 'MACD_Lag1', 'Stoch_K_Lag1', 'ATR_Lag1', 'ROC_Lag1',
        'Volume', 'Volume_Lag1',
        'Hour', 'DayOfWeek', 'DayOfMonth', 'Month'
    ]
    X = df_features[feature_columns]
    y = (df_features['Target'] > df_features['Close']).astype(int)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    logger.info(f"X shape: {X.shape}, y distribution: {y.value_counts().to_dict()}")
    return X, y, scaler, df_features

def train_model():
    try:
        X, y, scaler, df_original = prepare_data()
    except Exception as e:
        logger.error(f"Data preparation error, cannot train model: {e}")
        return

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    logger.info(f"Train/Validation set size: {len(X_train_val)}, Test set size: {len(X_test)}")

    neg_count = y_train_val.value_counts().get(0, 0)
    pos_count = y_train_val.value_counts().get(1, 0)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info(f"Class distribution in training data: Neg={neg_count}, Pos={pos_count}. scale_pos_weight={scale_pos_weight:.2f}")


    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.5),
            'random_state': 42,
            'force_col_wise': True,
            'verbose': -1,
            'n_jobs': -1,
            'scale_pos_weight': scale_pos_weight,
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

    best_model = LGBMClassifier(**best_params, random_state=42, force_col_wise=True, verbose=-1, scale_pos_weight=scale_pos_weight)
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

    if not df_original.empty:
        generate_signal(best_model, scaler, df_original.iloc[-1:], df_original.index[-1])
    else:
        logger.warning("No data to generate signal after training.")

def generate_signal(model, scaler, latest_data, last_index):
    try:
        if latest_data.empty:
            logger.warning("No latest data to generate signal.")
            return

        feature_columns = [
            'Open', 'High', 'Low', 'Close', 
            'RSI', 'MA20', 'BB_Up', 'BB_Low', 'MACD', 'MACD_Sig',
            'Stoch_K', 'Stoch_D', 'ATR', 'ROC',
            'Close_Lag1', 'RSI_Lag1', 'MACD_Lag1', 'Stoch_K_Lag1', 'ATR_Lag1', 'ROC_Lag1',
            'Volume', 'Volume_Lag1',
            'Hour', 'DayOfWeek', 'DayOfMonth', 'Month'
        ]
        features_for_scaling = latest_data[feature_columns]
        
        latest_data_scaled = scaler.transform(features_for_scaling)
        latest_data_scaled = pd.DataFrame(latest_data_scaled, columns=features_for_scaling.columns, index=latest_data.index)
        
        prediction_proba = model.predict_proba(latest_data_scaled)[0]
        
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
                # Для 15-минутных данных нет необходимости пропускать выходные
                df_latest_full = yf.download("EURUSD=X", interval="15m", period="7d", end=end_date) # <-- ИЗМЕНЕНО: период для получения последних данных
                if isinstance(df_latest_full.columns, pd.MultiIndex):
                    df_latest_full.columns = [col[0] for col in df_latest_full.columns]
                
                if 'Volume' in df_latest_full.columns and df_latest_full['Volume'].sum() == 0:
                    logger.warning("Volume data for latest download is present but all values are zero.")
                elif 'Volume' not in df_latest_full.columns:
                    logger.warning("Volume column not found in latest downloaded data. It will not be used as a feature.")
                    df_latest_full['Volume'] = 0

                df_latest_full['Close'] = df_latest_full['Close'].fillna(method='ffill').fillna(method='bfill')
                
                df_latest_full['RSI'] = compute_rsi(df_latest_full['Close'])
                df_latest_full['MA20'] = df_latest_full['Close'].rolling(window=20).mean()
                df_latest_full['BB_Up'], df_latest_full['BB_Low'] = compute_bollinger_bands(df_latest_full['Close'])
                df_latest_full['MACD'], df_latest_full['MACD_Sig'] = compute_macd(df_latest_full['Close'])
                
                df_latest_full['Stoch_K'], df_latest_full['Stoch_D'] = compute_stochastic_oscillator(df_latest_full['High'], df_latest_full['Low'], df_latest_full['Close'])
                df_latest_full['ATR'] = compute_atr(df_latest_full['High'], df_latest_full['Low'], df_latest_full['Close'])
                df_latest_full['ROC'] = compute_roc(df_latest_full['Close'])

                df_latest_full['Close_Lag1'] = df_latest_full['Close'].shift(1)
                df_latest_full['RSI_Lag1'] = df_latest_full['RSI'].shift(1)
                df_latest_full['MACD_Lag1'] = df_latest_full['MACD'].shift(1)
                df_latest_full['Stoch_K_Lag1'] = df_latest_full['Stoch_K'].shift(1)
                df_latest_full['ATR_Lag1'] = df_latest_full['ATR'].shift(1)
                df_latest_full['ROC_Lag1'] = df_latest_full['ROC'].shift(1)
                df_latest_full['Volume_Lag1'] = df_latest_full['Volume'].shift(1)

                df_latest_full['Hour'] = df_latest_full.index.hour
                df_latest_full['DayOfWeek'] = df_latest_full.index.dayofweek
                df_latest_full['DayOfMonth'] = df_latest_full.index.day
                df_latest_full['Month'] = df_latest_full.index.month

                df_latest_full['PriceChange'] = df_latest_full['Close'].pct_change()
                df_latest_full = df_latest_full[df_latest_full['PriceChange'].abs() < 0.1]
                df_latest_full = df_latest_full.dropna()

                if not df_latest_full.empty:
                    generate_signal(model, scaler, df_latest_full.iloc[-1:], df_latest_full.index[-1])
                else:
                    logger.warning("Could not get enough latest data to generate signal from existing model.")
            except Exception as e:
                logger.error(f"Error loading model or generating signal from existing model: {e}")

    return {"status": "Bot is running"}

if __name__ == "__main__":
    uvicorn.run(app, host=config.UVICORN_HOST, port=config.UVICORN_PORT)
