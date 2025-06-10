import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import config

def fetch_data(symbol):
    try:
        ticker = config.TICKER_MAP[symbol]  # Используем TICKER_MAP
        df = yf.download(ticker, period=f"{config.HISTORY_LIMIT}d", interval=config.TIMEFRAME)
        if df.empty:
            raise ValueError(f"Данные для {ticker} пусты")
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high',
                          'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"[DEBUG] Загружены данные для {symbol} ({ticker}): {df.shape}")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        raise Exception(f"Ошибка загрузки данных для {symbol}: {e}")

def prepare_features(df, lookahead_days=4):
    try:
        steps = lookahead_days
        df['target'] = df['close'].shift(-steps)
        df['direction'] = (df['target'] > df['close']).astype(int)

        df['ma_short'] = df['close'].rolling(10).mean()
        df['ma_long'] = df['close'].rolling(50).mean()
        df['volatility'] = df['high'] - df['low']
        df['momentum'] = df['close'].diff(5)

        df.dropna(inplace=True)
        if df.empty:
            raise ValueError("После обработки данных DataFrame пуст")
        features = ['open', 'high', 'low', 'close', 'volume', 'ma_short', 'ma_long', 'volatility', 'momentum']
        X = df[features]
        y = df['direction']
        print(f"[DEBUG] Подготовлены признаки: X.shape={X.shape}, y.shape={y.shape}")
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
