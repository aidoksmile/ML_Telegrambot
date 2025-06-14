import yfinance as yf
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from fastapi import FastAPI
import uvicorn
from datetime import datetime, timedelta
from send_telegram import send_telegram_message
import time

app = FastAPI()

MODEL_PATH = "model.pkl"
ACCURACY_PATH = "accuracy.json"

# Параметры стратегии
HORIZON_PERIODS = 1  # 1 день вперед
LOOKBACK_PERIOD = "max"  # Максимальный период данных
MIN_DATA_ROWS = 100
TARGET_ACCURACY = 0.8
MIN_ACCURACY_FOR_SIGNAL = 0.5
MAX_TRAINING_TIME = 3600  # 1 час

def compute_rsi(data, periods=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / (loss + 1e-10)  # Избегаем деления на ноль
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
        if end_date.weekday() >= 5:
            end_date -= timedelta(days=end_date.weekday() - 4)
        df = yf.download("EURUSD=X", interval="1d", period=LOOKBACK_PERIOD, end=end_date, auto_adjust=False)
    except Exception as e:
        raise ValueError(f"Error downloading data: {str(e)}")

    if df.empty:
        raise ValueError("Empty data received.")

    print(f"Downloaded {len(df)} rows, start date: {df.index[0]}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Фильтрация аномалий
    df = df[(df['Close'] > 0) & (df['Open'] > 0) & (df['Volume'] > 0)]
    if df['Close'].isna().any():
        print("Filling NaN values in 'Close'...")
        df['Close'] = df['Close'].fillna(method='ffill').fillna(method='bfill')

    # Создание Target до расчёта индикаторов
    df['Target'] = df['Close'].shift(-HORIZON_PERIODS)
    print(f"Target NaN count before cleaning: {df['Target'].isna().sum()}")

    if 'Target' not in df.columns or df['Target'].isna().all():
        raise ValueError(f"Failed to create target column. Close shape: {df['Close'].shape}, NaN in Close: {df['Close'].isna().sum()}")

    # Индикаторы
    df['RSI'] = compute_rsi(df['Close'])
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_Up'], df['BB_Low'] = compute_bollinger_bands(df['Close'])
    df['Lag1'] = df['Close'].shift(1)
    df['MACD'], df['MACD_Sig'] = compute_macd(df['Close'])
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek

    # Фильтрация волатильности
    df['PriceChange'] = df['Close'].pct_change()
    df = df[df['PriceChange'].abs() < 0.1]

    initial_rows = len(df)
    df = df.dropna()
    print(f"After dropping NaNs: {len(df)} rows (dropped {initial_rows - len(df)})")

    if len(df) < MIN_DATA_ROWS:
        raise ValueError(f"Insufficient data: {len(df)} rows, required {MIN_DATA_ROWS}")

    X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MA20', 'BB_Up', 'BB_Low', 'Lag1', 'MACD', 'MACD_Sig', 'Hour', 'DayOfWeek']].copy()
    y = (df['Target'] > df['Close']).astype(int)

    # Проверка на inf и NaN
    X = X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
    if X.isna().any().any():
        raise ValueError("NaN values remain in X after cleaning.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if np.any(np.isinf(X_scaled)) or np.any(np.isnan(X_scaled)):
        raise ValueError("Scaler produced inf or NaN values.")

    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    print(f"X shape: {X.shape}, y classes: {y.value_counts().to_dict()}")
    return X, y, scaler

def train_model():
    try:
        X, y, scaler = prepare_data()
    except Exception as e:
        print(f"Data prep error: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    param_dist = {
        'n_estimators': [50, 100],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.7, 1.0],
        'min_child_samples': [10, 20],
        'min_gain_to_split': [0.0, 0.01]
    }

    best_acc = 0.0
    best_model = None
    attempt = 1
    start_training = time.time()

    while best_acc < TARGET_ACCURACY and (time.time() - start_training) < MAX_TRAINING_TIME:
        print(f"Attempt #{attempt}...")
        start_time = time.time()
        try:
            model = LGBMClassifier(random_state=42, force_col_wise=True, verbose=-1)
            search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=4, cv=3, 
                                       scoring='accuracy', n_jobs=-1, random_state=42, verbose=0)
            search.fit(X_train, y_train)
            
            best_model = search.best_estimator_
            preds = best_model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            elapsed_time = time.time() - start_time
            print(f"Accuracy: {acc:.2f}, params: {search.best_params_}, time: {elapsed_time:.2f} sec")

            if acc > best_acc:
                best_acc = acc
                best_model = search.best_estimator_

            if best_acc >= TARGET_ACCURACY:
                print(f"Target accuracy {best_acc:.2f} reached. Saving model.")
                break

        except Exception as e:
            print(f"Error in attempt #{attempt}: {str(e)}")
            attempt += 1
            continue

        attempt += 1
        print(f"Accuracy {best_acc:.2f} below target ({TARGET_ACCURACY}). Retrying...")

    if (time.time() - start_training) >= MAX_TRAINING_TIME:
        print(f"Training timed out after {MAX_TRAINING_TIME} seconds.")

    if best_acc < MIN_ACCURACY_FOR_SIGNAL:
        print(f"Accuracy {best_acc:.2f} below minimum ({MIN_ACCURACY_FOR_SIGNAL}). No signal generated.")
        return

    joblib.dump({'model': best_model, 'scaler': scaler}, MODEL_PATH)
    with open(ACCURACY_PATH, "w") as f:
        json.dump({"accuracy": best_acc, "last_trained": str(datetime.now()), 
                   "best_params": search.best_params_ if best_model else {}}, f)

    generate_signal(best_model, scaler, X.iloc[-1:], X.index[-1])

def generate_signal(model, scaler, latest_data, last_index):
    try:
        latest_data_scaled = scaler.transform(latest_data)
        latest_data_scaled = pd.DataFrame(latest_data_scaled, columns=latest_data.columns, index=latest_data.index)
        prediction = model.predict(latest_data_scaled)[0]
        current_price = latest_data['Close'].iloc[0]
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
        print(f"Signal generated: {msg}")
        send_telegram_message(msg)
    except Exception as e:
        print(f"Error generating signal: {str(e)}")

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
