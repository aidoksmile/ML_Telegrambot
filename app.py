import yfinance as yf
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score # Добавлена f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from fastapi import FastAPI
import uvicorn
from datetime import datetime, timedelta
# from send_telegram import send_telegram_message # Закомментировано, так как send_telegram_message не предоставлен
import time
import optuna
import logging # Импорт модуля логирования

# Импорт конфигурации
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

# Заглушка для send_telegram_message, если она не определена
def send_telegram_message(message):
    logger.info(f"Telegram message (mock): {message}")

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
        while end_date.weekday() >= 5: # Пропускаем выходные
            end_date -= timedelta(days=1)
        # Используем LOOKBACK_PERIOD из config.py
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

    df['RSI'] = compute_rsi(df['Close'])
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_Up'], df['BB_Low'] = compute_bollinger_bands(df['Close'])
    df['Lag1'] = df['Close'].shift(1)
    df['MACD'], df['MACD_Sig'] = compute_macd(df['Close'])
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['PriceChange'] = df['Close'].pct_change()
    # Внимание: фильтрация экстремальных изменений цен может отбросить важные данные.
    # Используйте осторожно и проверьте влияние на производительность.
    df = df[df['PriceChange'].abs() < 0.1] 

    initial_rows = len(df)
    # dropna() удаляет строки, где индикаторы не могли быть рассчитаны из-за недостатка данных
    # или где были NaN в исходных данных.
    df = df.dropna() 
    logger.info(f"After dropna: {len(df)} rows (dropped {initial_rows - len(df)})")

    if len(df) < MIN_DATA_ROWS:
        logger.error(f"Insufficient data: {len(df)} rows, required {MIN_DATA_ROWS}")
        raise ValueError(f"Insufficient data: {len(df)} rows, required {MIN_DATA_ROWS}")

    X = df[['Open', 'High', 'Low', 'Close', 'RSI', 'MA20', 'BB_Up', 'BB_Low',
            'Lag1', 'MACD', 'MACD_Sig', 'Hour', 'DayOfWeek']]
    y = (df['Target'] > df['Close']).astype(int) # 1 for BUY, 0 for SELL/HOLD

    X = X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    logger.info(f"X shape: {X.shape}, y distribution: {y.value_counts().to_dict()}")
    return X, y, scaler

def train_model():
    try:
        X, y, scaler = prepare_data()
    except Exception as e:
        logger.error(f"Data preparation error, cannot train model: {e}")
        return

    # Разделение на обучающую/валидационную (для Optuna) и тестовую (для финальной оценки) выборки
    # Тестовая выборка - последние 20% данных
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
        }

        model = LGBMClassifier(**params)
        
        # Time Series Cross-Validation для оценки гиперпараметров
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_TS_CV)
        f1_scores = []

        for train_index, val_index in tscv.split(X_train_val):
            X_fold_train, X_fold_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
            y_fold_train, y_fold_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

            model.fit(X_fold_train, y_fold_train)
            preds = model.predict(X_fold_val)
            f1_scores.append(f1_score(y_fold_val, preds)) # Используем f1_score

        avg_f1 = np.mean(f1_scores)
        
        # Добавляем отчетность для прунера
        trial.report(avg_f1, trial.number)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return avg_f1

    logger.info("🔍 Starting Optuna hyperparameter search...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, timeout=MAX_TRAINING_TIME)

    best_params = study.best_params
    best_f1_score = study.best_value # Теперь это F1-score

    logger.info(f"✅ Optuna best F1-score: {best_f1_score:.4f}")
    logger.info(f"📋 Best params: {best_params}")

    if best_f1_score < MIN_ACCURACY_FOR_SIGNAL: # Используем F1-score для порога
        logger.warning(f"❌ F1-score {best_f1_score:.2f} too low. No model saved.")
        return

    # Обучаем финальную модель на всей обучающей/валидационной выборке с лучшими параметрами
    best_model = LGBMClassifier(**best_params, random_state=42, force_col_wise=True, verbose=-1)
    best_model.fit(X_train_val, y_train_val)

    # Оцениваем финальную модель на тестовой выборке
    test_preds = best_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, test_preds) # Можно оставить accuracy для общей оценки
    final_f1_score = f1_score(y_test, test_preds)
    logger.info(f"Final model performance on TEST set: Accuracy={final_accuracy:.4f}, F1-score={final_f1_score:.4f}")

    joblib.dump({'model': best_model, 'scaler': scaler}, MODEL_PATH)
    with open(ACCURACY_PATH, "w") as f:
        json.dump({
            "accuracy": final_accuracy, # Сохраняем финальную точность на тестовой выборке
            "f1_score": final_f1_score, # Сохраняем финальный F1-score на тестовой выборке
            "last_trained": str(datetime.now()),
            "best_params": best_params
        }, f)

    # Генерируем сигнал на основе последней доступной точки данных
    if not X.empty:
        generate_signal(best_model, scaler, X.iloc[-1:], X.index[-1])
    else:
        logger.warning("No data to generate signal after training.")

def generate_signal(model, scaler, latest_data, last_index):
    try:
        if latest_data.empty:
            logger.warning("No latest data to generate signal.")
            return

        latest_data_scaled = scaler.transform(latest_data)
        latest_data_scaled = pd.DataFrame(latest_data_scaled, columns=latest_data.columns, index=latest_data.index)
        
        # Получаем вероятности предсказания
        prediction_proba = model.predict_proba(latest_data_scaled)[0]
        
        # Определяем сигнал на основе порога вероятности
        signal_type = "HOLD"
        current_price = latest_data['Close'].iloc[0]
        stop_loss = None
        take_profit = None

        if prediction_proba[1] >= PREDICTION_PROB_THRESHOLD: # Вероятность класса 1 (BUY)
            signal_type = "BUY"
            stop_loss = current_price * 0.99 # Пример: 1% ниже
            take_profit = current_price * 1.015 # Пример: 1.5% выше
        elif prediction_proba[0] >= PREDICTION_PROB_THRESHOLD: # Вероятность класса 0 (SELL)
            signal_type = "SELL"
            stop_loss = current_price * 1.01 # Пример: 1% выше
            take_profit = current_price * 0.985 # Пример: 1.5% ниже
        
        # Комментарий: Фиксированные SL/TP могут быть неоптимальными.
        # Рассмотрите динамический расчет на основе волатильности (например, ATR).

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
        
        # Используем F1-score для проверки необходимости переобучения, если он есть
        current_metric = data.get("f1_score", data.get("accuracy", 0.0)) # Предпочитаем f1_score
        
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
                df_latest = yf.download("EURUSD=X", interval="1d", period="30d", end=end_date) 
                if isinstance(df_latest.columns, pd.MultiIndex):
                    df_latest.columns = [col[0] for col in df_latest.columns]
                df_latest['Close'] = df_latest['Close'].fillna(method='ffill').fillna(method='bfill')
                
                df_latest['RSI'] = compute_rsi(df_latest['Close'])
                df_latest['MA20'] = df_latest['Close'].rolling(window=20).mean()
                df_latest['BB_Up'], df_latest['BB_Low'] = compute_bollinger_bands(df_latest['Close'])
                df_latest['Lag1'] = df_latest['Close'].shift(1)
                df_latest['MACD'], df_latest['MACD_Sig'] = compute_macd(df_latest['Close'])
                df_latest['Hour'] = df_latest.index.hour
                df_latest['DayOfWeek'] = df_latest.index.dayofweek
                df_latest['PriceChange'] = df_latest['Close'].pct_change()
                df_latest = df_latest[df_latest['PriceChange'].abs() < 0.1]
                df_latest = df_latest.dropna()

                if not df_latest.empty:
                    latest_features = df_latest[['Open', 'High', 'Low', 'Close', 'RSI', 'MA20', 'BB_Up', 'BB_Low',
                                                 'Lag1', 'MACD', 'MACD_Sig', 'Hour', 'DayOfWeek']].iloc[-1:]
                    generate_signal(model, scaler, latest_features, latest_features.index[-1])
                else:
                    logger.warning("Could not get enough latest data to generate signal from existing model.")
            except Exception as e:
                logger.error(f"Error loading model or generating signal from existing model: {e}")

    return {"status": "Bot is running"}

if __name__ == "__main__":
    uvicorn.run(app, host=config.UVICORN_HOST, port=config.UVICORN_PORT)
