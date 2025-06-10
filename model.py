import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import config
import requests
import time

def fetch_data(symbol):
    try:
        ticker = config.TICKER_MAP[symbol]
        timeframe = config.TIMEFRAME.get(symbol, "MINUTE_15")
        session_url = f"{config.BASE_URL}/api/v1/session"
        prices_url = f"{config.BASE_URL}/api/v1/prices/{ticker}"

        # Авторизация
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "identifier": config.CAPITAL_EMAIL,
            "password": config.CAPITAL_PASSWORD,
            "encryptedPassword": False
        }
        for attempt in range(3):
            try:
                response = requests.post(session_url, headers=headers, json=payload)
                response.raise_for_status()
                session_data = response.json()
                cst = response.headers.get("CST")
                security_token = response.headers.get("X-SECURITY-TOKEN")
                if not cst or not security_token:
                    raise ValueError("Не удалось получить CST или X-SECURITY-TOKEN")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"[WARNING] Попытка авторизации {attempt + 1} не удалась: {e}, повтор через 5 секунд")
                    time.sleep(5)
                else:
                    raise ValueError(f"Не удалось авторизоваться после 3 попыток: {e}")

        # Запрос исторических данных
        headers = {
            "CST": cst,
            "X-SECURITY-TOKEN": security_token
        }
        params = {
            "resolution": timeframe,
            "maxReturn": config.HISTORY_LIMIT
        }
        for attempt in range(3):
            try:
                response = requests.get(prices_url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                if "prices" not in data:
                    raise ValueError(f"Данные для {ticker} отсутствуют: {data.get('error', 'No data')}")
                df = pd.DataFrame(data["prices"])
                df = df.rename(columns={
                    "snapshotTimeUTC": "timestamp",
                    "openPrice": "open",
                    "highPrice": "high",
                    "lowPrice": "low",
                    "closePrice": "close",
                    "volume": "volume"
                })
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col].apply(lambda x: x.get("bid") if isinstance(x, dict) else x), errors="coerce")
                df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce")
                df = df.sort_values("timestamp").tail(config.HISTORY_LIMIT)
                if df.empty or len(df) < 50:
                    raise ValueError(f"Недостаточно данных для {ticker}: {len(df)} строк")
                required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Отсутствуют столбцы: {required_cols}, получено: {df.columns.tolist()}")
                print(f"[DEBUG] Загружены данные для {symbol} ({ticker}): shape={df.shape}, columns={df.columns.tolist()}, timeframe={timeframe}")
                return df[required_cols]
            except Exception as e:
                if attempt < 2:
                    print(f"[WARNING] Попытка {attempt + 1} не удалась для {ticker}: {e}, повтор через 5 секунд")
                    time.sleep(5)
                else:
                    raise
        raise ValueError(f"Не удалось загрузить данные для {ticker} после 3 попыток")
    except Exception as e:
        raise Exception(f"Ошибка загрузки данных для {symbol}: {e}")

def prepare_features(df, lookahead_days=4):
    try:
        df = df.copy()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Отсутствуют необходимые столбцы: {required_cols}, получено: {df.columns.tolist()}")

        df = df.reset_index(drop=True)
        print(f"[DEBUG] Исходный DataFrame: shape={df.shape}, columns={df.columns.tolist()}")

        steps = lookahead_days
        if len(df) < steps + 50:
            raise ValueError(f"Недостаточно данных: {len(df)} строк, требуется минимум {steps + 50}")

        df['target'] = df['close'].shift(-steps)
        if df['target'].isna().all():
            raise ValueError(f"Столбец 'target' содержит только NaN, возможно, недостаточно данных для сдвига на {steps} шагов")

        df = df.dropna(subset=['target', 'close'])
        if df.empty:
            raise ValueError("DataFrame пуст после удаления NaN в 'target' и 'close'")

        df['direction'] = (df['target'] > df['close']).astype(int)

        df['ma_short'] = df['close'].rolling(window=10, min_periods=10).mean()
        df['ma_long'] = df['close'].rolling(window=50, min_periods=50).mean()
        df['volatility'] = df['high'] - df['low']
        df['momentum'] = df['close'].diff(5)

        df = df.dropna()
        if df.empty:
            raise ValueError("DataFrame пуст после удаления NaN для признаков")

        features = ['open', 'high', 'low', 'close', 'volume', 'ma_short', 'ma_long', 'volatility', 'momentum']
        X = df[features]
        y = df['direction']

        print(f"[DEBUG] Признаки подготовлены: X.shape={X.shape}, y.shape={y.shape}")
        return X, y
    except Exception as e:
        raise Exception(f"Ошибка подготовки признаков: {e}")

def optimize_model_params(X, y):
    try:
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=param_grid,
            scoring='accuracy',
            cv=3,
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"[Оптимизация] Лучшие параметры: {best_params}")
        print(f"[Оптимизация] Точность на кросс-валидации: {best_score * 100:.2f}%")
        return grid_search.best_estimator_, best_score
    except Exception as e:
        raise Exception(f"Ошибка оптимизации модели: {e}")

def train_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = RandomForestClassifier(n_estimators=50)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Точность модели: {accuracy * 100:.2f}%")

        if accuracy >= 0.8:
            print("✅ Модель прошла порог точности")
            return model

        print("⚠️ Точность ниже 80%, запуск оптимизации...")
        optimized_model, _ = optimize_model_params(X, y)
        acc = accuracy_score(y_test, optimized_model.predict(X_test))
        print(f"✅ Новая точность после оптимизации: {acc * 100:.2f}%")

        if acc < 0.8:
            raise ValueError(f"Не удалось достичь точности 80% (текущая: {acc * 100:.2f}%)")

        return optimized_model
    except Exception as e:
        raise Exception(f"Ошибка обучения модели: {e}")
