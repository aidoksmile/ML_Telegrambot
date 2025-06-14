import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
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
HORIZON_DAYS = 1  # Прогноз на 1 день вперед (~96 свечей по 15 мин)
LOOKBACK_PERIOD = "60d"  # История для обучения — 60 дней
MIN_DATA_ROWS = 100  # Минимальное количество строк для обучения
TARGET_ACCURACY = 0.8  # Целевая точность
MAX_RETRAIN_ATTEMPTS = 3  # Максимальное количество попыток перетренировки (уменьшено для скорости)
MIN_ACCURACY_FOR_SIGNAL = 0.5  # Минимальная точность для генерации сигнала

def prepare_data():
    print("Загрузка данных из Yahoo Finance...")
    try:
        end_date = datetime.now()
        if end_date.weekday() >= 5:  # Saturday (5) or Sunday (6)
            end_date -= timedelta(days=end_date.weekday() - 4)  # Go to last Friday
        df = yf.download("EURUSD=X", interval="15m", period=LOOKBACK_PERIOD, end=end_date)
    except Exception as e:
        raise ValueError(f"Ошибка при загрузке данных из Yahoo Finance: {str(e)}")

    if df.empty or len(df) == 0:
        raise ValueError("Данные из Yahoo Finance пусты или не содержат строк.")

    print(f"Downloaded {len(df)} rows.")
    print("Column structure:\n", df.columns)

    if isinstance(df.columns, pd.MultiIndex):
        print("Multi-index columns detected. Flattening to single-level columns.")
        df.columns = [col[0] for col in df.columns]

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Отсутствуют необходимые столбцы: {missing_columns}")

    print("Initial data sample:\n", df.head())
    print("Missing values:\n", df.isna().sum())

    if df['Close'].isna().all():
        raise ValueError("Столбец 'Close' содержит только NaN значения.")
    if df['Close'].isna().any():
        print("Warning: Обнаружены пропущенные значения в столбце 'Close'. Заполняем их.")
        df['Close'] = df['Close'].fillna(method='ffill').fillna(method='bfill')

    try:
        df['target'] = df['Close'].shift(-int(HORIZON_DAYS * 96))
    except Exception as e:
        raise ValueError(f"Ошибка при создании столбца 'target': {str(e)}")

    if 'target' not in df.columns:
        raise ValueError("Не удалось создать столбец 'target'.")

    if df['target'].isna().all():
        raise ValueError("Столбец 'target' содержит только NaN значения, возможно из-за недостатка данных.")

    print("Target column sample:\n", df[['Close', 'target']].head())
    print("Target NaN count:", df['target'].isna().sum())

    initial_rows = len(df)
    df = df.dropna(subset=['target'])
    print(f"After dropping NaN target rows: {len(df)} (removed {initial_rows - len(df)} rows)")

    if df.empty or len(df) == 0:
        raise ValueError("DataFrame пуст после удаления строк с NaN в 'target'.")

    if len(df) < MIN_DATA_ROWS:
        raise ValueError(f"Недостаточно данных для обучения модели. Available rows: {len(df)}, required: {MIN_DATA_ROWS}")

    try:
        X = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        y = (df['target'] > df['Close']).astype(int)
    except Exception as e:
        raise ValueError(f"Ошибка при создании X или y: {str(e)}. Проверьте выравнивание 'target' и 'Close'.")

    if len(X) != len(y):
        raise ValueError(f"X и y не выровнены: X имеет {len(X)} строк, y имеет {len(y)} строк")

    print("Final X shape:", X.shape)
    print("Final y value counts:", y.value_counts().to_dict())

    return X, y

def train_model():
    try:
        X, y = prepare_data()
    except Exception as e:
        print(f"Ошибка подготовки данных: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define hyperparameter grid for RandomizedSearchCV
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    best_acc = 0.0
    best_model = None
    attempt = 1

    while best_acc < TARGET_ACCURACY and attempt <= MAX_RETRAIN_ATTEMPTS:
        print(f"Попытка обучения модели #{attempt}...")
        start_time = time.time()
        try:
            model = RandomForestClassifier(random_state=42)
            # Use RandomizedSearchCV for faster search
            search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, 
                                       scoring='accuracy', n_jobs=-1, random_state=42)
            search.fit(X_train, y_train)
            
            best_model = search.best_estimator_
            preds = best_model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            elapsed_time = time.time() - start_time
            print(f"Попытка #{attempt} - Лучшая точность: {acc:.2f}, "
                  f"Лучшие параметры: {search.best_params_}, Время: {elapsed_time:.2f} сек")

            if acc > best_acc:
                best_acc = acc
                best_model = search.best_estimator_

            if best_acc >= TARGET_ACCURACY:
                print(f"Достигнута целевая точность {best_acc:.2f}. Сохраняем модель.")
                break

        except Exception as e:
            print(f"Ошибка в попытке #{attempt}: {str(e)}")
            attempt += 1
            continue

        attempt += 1
        print(f"Точность {best_acc:.2f} ниже целевой ({TARGET_ACCURACY}). Пробуем снова...")

    if best_acc < TARGET_ACCURACY:
        print(f"Не удалось достичь целевой точности {TARGET_ACCURACY} после {MAX_RETRAIN_ATTEMPTS} попыток. "
              f"Лучшая точность: {best_acc:.2f}.")

    if best_acc < MIN_ACCURACY_FOR_SIGNAL:
        print(f"Точность {best_acc:.2f} ниже минимальной ({MIN_ACCURACY_FOR_SIGNAL}). Сигнал не будет сгенерирован.")
        return

    # Save the best model
    joblib.dump(best_model, MODEL_PATH)

    # Save accuracy and training info
    with open(ACCURACY_PATH, "w") as f:
        json.dump({"accuracy": best_acc, "last_trained": str(datetime.now()), 
                   "best_params": search.best_params_ if best_model else {}}, f)

    # Generate signal with the best model
    generate_signal(best_model, X.iloc[-1:], X.index[-1])

def generate_signal(model, latest_data, last_index):
    prediction = model.predict(latest_data)[0]
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
    print(f"Генерация сигнала: {msg}")
    send_telegram_message(msg)

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
