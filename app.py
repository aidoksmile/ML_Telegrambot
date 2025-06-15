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
# from send_telegram import send_telegram_message # Закомментировано, так как send_telegram_message не предоставлен
import time
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
MAX_TRAINING_TIME = 3600 # Максимальное время обучения в секундах

# Заглушка для send_telegram_message, если она не определена
def send_telegram_message(message):
    print(f"Telegram message (mock): {message}")

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
    print("Downloading data...")
    try:
        end_date = datetime.now()
        while end_date.weekday() >= 5: # Пропускаем выходные
            end_date -= timedelta(days=1)
        df = yf.download("EURUSD=X", interval="1d", period=LOOKBACK_PERIOD, end=end_date)
    except Exception as e:
        raise ValueError(f"Error downloading data: {str(e)}")

    if df.empty:
        raise ValueError("Empty data received from Yahoo Finance.")

    print(f"Downloaded {len(df)} rows, from {df.index[0]} to {df.index[-1]}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df[(df['Close'] > 0) & (df['Open'] > 0)]
    print(f"After filtering Open/Close > 0: {len(df)} rows")

    df['Close'] = df['Close'].fillna(method='ffill').fillna(method='bfill')
    df['Target'] = df['Close'].shift(-HORIZON_PERIODS)

    if df['Target'].isna().all():
        raise ValueError("Target column contains only NaNs.")

    df['RSI'] = compute_rsi(df['Close'])
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_Up'], df['BB_Low'] = compute_bollinger_bands(df['Close'])
    df['Lag1'] = df['Close'].shift(1)
    df['MACD'], df['MACD_Sig'] = compute_macd(df['Close'])
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['PriceChange'] = df['Close'].pct_change()
    df = df[df['PriceChange'].abs() < 0.1] # Отфильтровываем экстремальные изменения цен

    initial_rows = len(df)
    df = df.dropna()
    print(f"After dropna: {len(df)} rows (dropped {initial_rows - len(df)})")

    if len(df) < MIN_DATA_ROWS:
        raise ValueError(f"Insufficient data: {len(df)} rows, required {MIN_DATA_ROWS}")

    X = df[['Open', 'High', 'Low', 'Close', 'RSI', 'MA20', 'BB_Up', 'BB_Low',
            'Lag1', 'MACD', 'MACD_Sig', 'Hour', 'DayOfWeek']]
    y = (df['Target'] > df['Close']).astype(int)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    print(f"X shape: {X.shape}, y distribution: {y.value_counts().to_dict()}")
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
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Добавляем отчетность для прунера
        trial.report(acc, trial.number)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return acc

    print("🔍 Starting Optuna hyperparameter search...")
    # Использование HyperbandPruner для более эффективного отсечения неперспективных испытаний
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, timeout=MAX_TRAINING_TIME)

    best_params = study.best_params
    best_acc = study.best_value

    print(f"✅ Optuna best accuracy: {best_acc:.4f}")
    print(f"📋 Best params: {best_params}")

    if best_acc < MIN_ACCURACY_FOR_SIGNAL:
        print(f"❌ Accuracy {best_acc:.2f} too low. No model saved.")
        return

    best_model = LGBMClassifier(**best_params, random_state=42, force_col_wise=True, verbose=-1)
    best_model.fit(X_train, y_train)

    joblib.dump({'model': best_model, 'scaler': scaler}, MODEL_PATH)
    with open(ACCURACY_PATH, "w") as f:
        json.dump({
            "accuracy": best_acc,
            "last_trained": str(datetime.now()),
            "best_params": best_params
        }, f)

    # Генерируем сигнал на основе последней доступной точки данных
    if not X.empty:
        generate_signal(best_model, scaler, X.iloc[-1:], X.index[-1])
    else:
        print("No data to generate signal.")

def generate_signal(model, scaler, latest_data, last_index):
    try:
        # Проверяем, что latest_data не пуста
        if latest_data.empty:
            print("No latest data to generate signal.")
            return

        latest_data_scaled = scaler.transform(latest_data)
        latest_data_scaled = pd.DataFrame(latest_data_scaled, columns=latest_data.columns, index=latest_data.index)
        prediction = model.predict(latest_data_scaled)[0]
        current_price = latest_data['Close'].iloc[0]
        
        # Расчет Stop Loss и Take Profit
        # Для BUY (prediction == 1): SL ниже, TP выше
        # Для SELL (prediction == 0): SL выше, TP ниже
        stop_loss = current_price * (0.99 if prediction == 1 else 1.01)
        take_profit = current_price * (1.015 if prediction == 1 else 0.985)

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
    except Exception as e:
        print(f"Signal generation error: {e}")

@app.get("/")
async def root():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ACCURACY_PATH):
        print("Model or accuracy file not found. Training new model.")
        train_model()
    else:
        with open(ACCURACY_PATH, "r") as f:
            data = json.load(f)
        last_trained = datetime.fromisoformat(data["last_trained'])
        # Переобучаем модель, если прошло более 1 дня или точность ниже целевой
        if (datetime.now() - last_trained).days >= 1 or data["accuracy"] < TARGET_ACCURACY:
            print(f"Model needs retraining. Last trained: {last_trained}, Accuracy: {data['accuracy']:.2f}")
            train_model()
        else:
            print(f"Model is up to date. Last trained: {last_trained}, Accuracy: {data['accuracy']:.2f}")
            # Если модель актуальна, загружаем ее и генерируем сигнал с последними данными
            try:
                model_data = joblib.load(MODEL_PATH)
                model = model_data['model']
                scaler = model_data['scaler']
                # Получаем последние данные для генерации сигнала
                end_date = datetime.now()
                while end_date.weekday() >= 5:
                    end_date -= timedelta(days=1)
                df_latest = yf.download("EURUSD=X", interval="1d", period="5d", end=end_date) # Загружаем немного больше данных
                if isinstance(df_latest.columns, pd.MultiIndex):
                    df_latest.columns = [col[0] for col in df_latest.columns]
                df_latest['Close'] = df_latest['Close'].fillna(method='ffill').fillna(method='bfill')
                
                # Пересчитываем признаки для последних данных
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
                    print("Could not get enough latest data to generate signal.")
            except Exception as e:
                print(f"Error loading model or generating signal: {e}")

    return {"status": "Bot is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
