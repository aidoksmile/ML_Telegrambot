import yfinance as yf
import pandas as pd
import numpy as np
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
from optuna.integration import TFKerasPruningCallback

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

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
OPTUNA_STORAGE_URL = config.OPTUNA_STORAGE_URL
OPTUNA_STUDY_NAME = config.OPTUNA_STUDY_NAME
SEQUENCE_LENGTH = config.SEQUENCE_LENGTH
NN_EPOCHS = config.NN_EPOCHS
NN_BATCH_SIZE = config.NN_BATCH_SIZE
NN_PATIENCE = config.NN_PATIENCE

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
    roc = ((data - data.shift(period)) / data.shift(period)) * 100
    return roc

def create_sequences(features, target, sequence_length, horizon_periods):
    """
    Создает последовательности признаков и соответствующие целевые значения.
    
    Args:
        features (pd.DataFrame): DataFrame с признаками.
        target (pd.Series): Series с целевой переменной.
        sequence_length (int): Длина входной последовательности для NN.
        horizon_periods (int): Горизонт предсказания.
        
    Returns:
        tuple: (np.array X_seq, np.array y_seq)
    """
    X_seq, y_seq = [], []
    for i in range(len(features) - sequence_length - horizon_periods + 1):
        X_seq.append(features.iloc[i:(i + sequence_length)].values)
        y_seq.append(target.iloc[i + sequence_length + horizon_periods - 1])
        
    return np.array(X_seq), np.array(y_seq)


def prepare_data():
    logger.info("Downloading data...")
    try:
        end_date = datetime.now()
        df = yf.download("EURUSD=X", interval="15m", period=LOOKBACK_PERIOD, end=end_date)
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        raise ValueError(f"Error downloading data: {str(e)}")

    if df.empty:
        logger.error("Empty data received from Yahoo Finance.")
        raise ValueError("Empty data received from Yahoo Finance.")

    logger.info(f"Downloaded {len(df)} rows, from {df.index[0]} to {df.index[-1]} for 15m interval.")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df[(df['Close'] > 0) & (df['Open'] > 0)]
    logger.info(f"After filtering Open/Close > 0: {len(df)} rows")

    if 'Volume' in df.columns and df['Volume'].sum() == 0:
        logger.warning("Volume data is present but all values are zero. This is common for forex data from Yahoo Finance.")
    elif 'Volume' not in df.columns:
        logger.warning("Volume column not found in downloaded data. It will not be used as a feature.")
        df['Volume'] = 0

    df['Close'] = df['Close'].fillna(method='ffill').fillna(method='bfill')
    df['Target'] = df['Close'].shift(-HORIZON_PERIODS)

    if df['Target'].isna().all():
        logger.error("Target column contains only NaNs.")
        raise ValueError("Target column contains only NaNs.")

    df_features = df.copy() 

    df_features['RSI'] = compute_rsi(df_features['Close'])
    df_features['MA20'] = df_features['Close'].rolling(window=20).mean()
    df_features['BB_Up'], df_features['BB_Low'] = compute_bollinger_bands(df_features['Close'])
    df_features['MACD'], df_features['MACD_Sig'] = compute_macd(df_features['Close'])
    
    df_features['Stoch_K'], df_features['Stoch_D'] = compute_stochastic_oscillator(df_features['High'], df_features['Low'], df_features['Close'])
    df_features['ATR'] = compute_atr(df_features['High'], df_features['Low'], df_features['Close'])
    df_features['ROC'] = compute_roc(df_features['Close'])

    df_features['Close_Lag1'] = df_features['Close'].shift(1)
    df_features['RSI_Lag1'] = df_features['RSI'].shift(1)
    df_features['MACD_Lag1'] = df_features['MACD'].shift(1)
    df_features['Stoch_K_Lag1'] = df_features['Stoch_K'].shift(1)
    df_features['ATR_Lag1'] = df_features['ATR'].shift(1)
    df_features['ROC_Lag1'] = df_features['ROC'].shift(1)
    df_features['Volume_Lag1'] = df_features['Volume'].shift(1)

    df_features['Hour'] = df_features.index.hour
    df_features['DayOfWeek'] = df_features.index.dayofweek
    df_features['DayOfMonth'] = df_features.index.day
    df_features['Month'] = df_features.index.month

    df_features['PriceChange'] = df_features['Close'].pct_change()
    df_features = df_features[df_features['PriceChange'].abs() < 0.1] 

    initial_rows = len(df_features)
    df_features = df_features.dropna() 
    logger.info(f"After dropna: {len(df_features)} rows (dropped {initial_rows - len(df_features)})")

    required_rows = SEQUENCE_LENGTH + HORIZON_PERIODS
    if len(df_features) < required_rows:
        logger.error(f"Insufficient data: {len(df_features)} rows, required at least {required_rows} for sequences and horizon.")
        raise ValueError(f"Insufficient data: {len(df_features)} rows, required at least {required_rows}.")
    
    df_features = df_features.dropna(subset=['Target'])

    feature_columns = [
        'Open', 'High', 'Low', 'Close', 
        'RSI', 'MA20', 'BB_Up', 'BB_Low', 'MACD', 'MACD_Sig',
        'Stoch_K', 'Stoch_D', 'ATR', 'ROC',
        'Close_Lag1', 'RSI_Lag1', 'MACD_Lag1', 'Stoch_K_Lag1', 'ATR_Lag1', 'ROC_Lag1',
        'Volume', 'Volume_Lag1',
        'Hour', 'DayOfWeek', 'DayOfMonth', 'Month'
    ]
    X_raw = df_features[feature_columns]
    y_raw = (df_features['Target'] > df_features['Close']).astype(int)

    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_raw.columns, index=X_raw.index)

    X_seq, y_seq = create_sequences(X_scaled_df, y_raw, SEQUENCE_LENGTH, HORIZON_PERIODS)

    logger.info(f"X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}, y_seq distribution: {pd.Series(y_seq).value_counts().to_dict()}")
    return X_seq, y_seq, scaler, df_features

def train_model():
    try:
        X_seq, y_seq, scaler, df_original = prepare_data()
    except Exception as e:
        logger.error(f"Data preparation error, cannot train model: {e}")
        return

    if len(X_seq) != len(y_seq):
        raise ValueError("X_seq and y_seq must have the same number of samples.")

    test_size_ratio = 0.2
    split_index = int(len(X_seq) * (1 - test_size_ratio))
    
    X_train_val, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train_val, y_test = y_seq[:split_index], y_seq[split_index:]

    logger.info(f"Train/Validation set size: {len(X_train_val)}, Test set size: {len(X_test)}")

    neg_count = pd.Series(y_train_val).value_counts().get(0, 0)
    pos_count = pd.Series(y_train_val).value_counts().get(1, 0)
    class_weight = {0: 1.0, 1: neg_count / pos_count if pos_count > 0 else 1.0}
    logger.info(f"Class distribution in training data: Neg={neg_count}, Pos={pos_count}. Class weights={class_weight}")

    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)
        n_units = trial.suggest_int('n_units', 32, 256, step=32)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])

        model = keras.Sequential()
        model.add(layers.Flatten(input_shape=(SEQUENCE_LENGTH, X_train_val.shape[2])))
        
        for i in range(n_layers):
            model.add(layers.Dense(n_units, activation=activation))
            model.add(layers.Dropout(dropout_rate))
        
        model.add(layers.Dense(1, activation='sigmoid'))

        optimizer = None
        if optimizer_name == 'Adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'RMSprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.F1Score(average='weighted', name='f1_score')])

        tscv = TimeSeriesSplit(n_splits=N_SPLITS_TS_CV)
        f1_scores = []

        for fold, (train_index, val_index) in enumerate(tscv.split(X_train_val)):
            X_fold_train, X_fold_val = X_train_val[train_index], X_train_val[val_index]
            y_fold_train, y_fold_val = y_train_val[train_index], y_train_val[val_index]

            # --- ИСПРАВЛЕНИЕ: Изменение формы y_fold_train и y_fold_val ---
            y_fold_train_reshaped = np.expand_dims(y_fold_train, axis=-1)
            y_fold_val_reshaped = np.expand_dims(y_fold_val, axis=-1)

            callbacks = [
                TFKerasPruningCallback(trial, 'val_f1_score'),
                EarlyStopping(monitor='val_f1_score', patience=NN_PATIENCE, mode='max', restore_best_weights=True)
            ]

            history = model.fit(X_fold_train, y_fold_train_reshaped, # Используем измененную форму
                                epochs=NN_EPOCHS,
                                batch_size=NN_BATCH_SIZE,
                                validation_data=(X_fold_val, y_fold_val_reshaped), # Используем измененную форму
                                callbacks=callbacks,
                                verbose=0,
                                class_weight=class_weight)

            best_val_f1 = max(history.history['val_f1_score'])
            f1_scores.append(best_val_f1)

        avg_f1 = np.mean(f1_scores)
        
        trial.report(avg_f1, trial.number)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return avg_f1

    logger.info("🔍 Starting Optuna hyperparameter search for Neural Network...")
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

    final_model = keras.Sequential()
    final_model.add(layers.Flatten(input_shape=(SEQUENCE_LENGTH, X_train_val.shape[2]))) # ИСПРАВЛЕНО: input_input_shape -> input_shape
    for i in range(best_params['n_layers']):
        final_model.add(layers.Dense(best_params['n_units'], activation=best_params['activation']))
        final_model.add(layers.Dropout(best_params['dropout_rate']))
    final_model.add(layers.Dense(1, activation='sigmoid'))

    optimizer = None
    if best_params['optimizer'] == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    elif best_params['optimizer'] == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=best_params['learning_rate'])

    final_model.compile(optimizer=optimizer,
                        loss='binary_crossentropy',
                        metrics=['accuracy', tf.keras.metrics.F1Score(average='weighted', name='f1_score')])

    callbacks_final = [
        EarlyStopping(monitor='val_f1_score', patience=NN_PATIENCE * 2, mode='max', restore_best_weights=True)
    ]

    final_train_split_index = int(len(X_train_val) * 0.8)
    X_final_train, X_final_val = X_train_val[:final_train_split_index], X_train_val[final_train_split_index:]
    y_final_train, y_final_val = y_train_val[:final_train_split_index], y_train_val[final_train_split_index:]

    # --- ИСПРАВЛЕНИЕ: Изменение формы y_final_train и y_final_val ---
    y_final_train_reshaped = np.expand_dims(y_final_train, axis=-1)
    y_final_val_reshaped = np.expand_dims(y_final_val, axis=-1)

    logger.info(f"Final model training on {len(X_final_train)} samples, validating on {len(X_final_val)} samples.")
    final_model.fit(X_final_train, y_final_train_reshaped, # Используем измененную форму
                    epochs=NN_EPOCHS,
                    batch_size=NN_BATCH_SIZE,
                    validation_data=(X_final_val, y_final_val_reshaped), # Используем измененную форму
                    callbacks=callbacks_final,
                    verbose=1,
                    class_weight=class_weight)

    # --- ИСПРАВЛЕНИЕ: Изменение формы y_test ---
    y_test_reshaped = np.expand_dims(y_test, axis=-1)
    test_loss, test_accuracy, test_f1_score = final_model.evaluate(X_test, y_test_reshaped, verbose=0) # Используем измененную форму
    logger.info(f"Final model performance on TEST set: Accuracy={test_accuracy:.4f}, F1-score={test_f1_score:.4f}")

    final_model.save(MODEL_PATH)
    joblib.dump({'scaler': scaler}, 'scaler.pkl')

    with open(ACCURACY_PATH, "w") as f:
        json.dump({
            "accuracy": float(test_accuracy),
            "f1_score": float(test_f1_score),
            "last_trained": str(datetime.now()),
            "best_params": best_params
        }, f)

    if X_seq.shape[0] > 0:
        latest_sequence_scaled = X_seq[-1:] 
        latest_original_data_point = df_original.iloc[-1:]

        generate_signal(final_model, scaler, latest_sequence_scaled, latest_original_data_point)
    else:
        logger.warning("No data to generate signal after training.")


def generate_signal(model, scaler, latest_sequence_scaled, latest_original_data_point):
    try:
        if latest_sequence_scaled.shape[0] == 0:
            logger.warning("No latest sequence data to generate signal.")
            return
        if latest_original_data_point.empty:
            logger.warning("No latest original data point to get current price.")
            return

        prediction_proba = model.predict(latest_sequence_scaled)[0]
        
        current_price = latest_original_data_point['Close'].iloc[0] 
        
        signal_type = "HOLD"
        stop_loss = None
        take_profit = None

        buy_probability = prediction_proba[0] 
        sell_probability = 1 - prediction_proba[0] 

        if buy_probability >= PREDICTION_PROB_THRESHOLD:
            signal_type = "BUY"
            stop_loss = current_price * 0.99
            take_profit = current_price * 1.015
        elif sell_probability >= PREDICTION_PROB_THRESHOLD:
            signal_type = "SELL"
            stop_loss = current_price * 1.01
            take_profit = current_price * 0.985
        
        signal = {
            "time": str(datetime.now()),
            "price": round(current_price, 5),
            "signal": signal_type,
            "buy_proba": round(buy_probability, 4),
            "sell_proba": round(sell_probability, 4),
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
                model = keras.models.load_model(MODEL_PATH)
                scaler_data = joblib.load('scaler.pkl')
                scaler = scaler_data['scaler']
                
                end_date = datetime.now()
                df_latest_full = yf.download("EURUSD=X", interval="15m", period="7d", end=end_date) 
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

                if len(df_latest_full) >= SEQUENCE_LENGTH:
                    feature_columns = [
                        'Open', 'High', 'Low', 'Close', 
                        'RSI', 'MA20', 'BB_Up', 'BB_Low', 'MACD', 'MACD_Sig',
                        'Stoch_K', 'Stoch_D', 'ATR', 'ROC',
                        'Close_Lag1', 'RSI_Lag1', 'MACD_Lag1', 'Stoch_K_Lag1', 'ATR_Lag1', 'ROC_Lag1',
                        'Volume', 'Volume_Lag1',
                        'Hour', 'DayOfWeek', 'DayOfMonth', 'Month'
                    ]
                    latest_features_raw = df_latest_full[feature_columns].iloc[-SEQUENCE_LENGTH:]
                    latest_sequence_scaled = scaler.transform(latest_features_raw)
                    latest_sequence_scaled = np.expand_dims(latest_sequence_scaled, axis=0)

                    latest_original_data_point = df_latest_full.iloc[-1:]

                    generate_signal(model, scaler, latest_sequence_scaled, latest_original_data_point)
                else:
                    logger.warning("Could not get enough latest data to generate signal from existing model (need at least SEQUENCE_LENGTH rows).")
            except Exception as e:
                logger.error(f"Error loading model or generating signal from existing model: {e}")

    return {"status": "Bot is running"}

if __name__ == "__main__":
    uvicorn.run(app, host=config.UVICORN_HOST, port=config.UVICORN_PORT)
