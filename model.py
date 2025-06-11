import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import config
import asyncio
from metaapi_cloud_sdk import MetaApi

# Настройка логирования
logging.basicConfig(
    filename="bot.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

async def fetch_data(symbol):
    logging.info(f"Начало загрузки данных для {symbol}")
    account = None
    api = None
    try:
        # Инициализация MetaApi
        for attempt in range(3):
            try:
                logging.debug(f"Попытка {attempt + 1} инициализации MetaApi")
                api = MetaApi(token=config.METAAPI_TOKEN)
                logging.info("MetaApi успешно инициализирован")
                break
            except Exception as e:
                if attempt < 2:
                    logging.warning(f"Ошибка инициализации MetaApi: {e}, повтор через 5 секунд")
                    await asyncio.sleep(5)
                else:
                    logging.error(f"Не удалось инициализировать MetaApi после 3 попыток: {e}")
                    raise ValueError(f"Не удалось инициализировать MetaApi: {e}")

        # Получение аккаунта
        try:
            logging.debug(f"Запрос аккаунта с ID {config.MT4_ACCOUNT_ID}")
            account = await api.metatrader_account_api.get_account(config.MT4_ACCOUNT_ID)
            logging.info(f"Аккаунт {config.MT4_ACCOUNT_ID} получен")
            await account.connect()
            logging.info(f"Подключение к аккаунту для {symbol} успешно")
        except Exception as e:
            logging.error(f"Ошибка подключения к аккаунту: {e}")
            raise Exception(f"Ошибка подключения к аккаунту для {symbol}: {e}")

        # Получение таймфрейма
        timeframe_map = {
            "15m": "15m",
        }
        timeframe = timeframe_map.get(config.TIMEFRAME.get(symbol, "15m"), "15m")
        logging.debug(f"Используемый таймфрейм: {timeframe} для {symbol}")

        # Запрос исторических данных
        for attempt in range(3):
            try:
                logging.debug(f"Попытка {attempt + 1} получения свечей для {symbol}")
                candles = await account.get_historical_candles(
                    symbol=symbol,
                    timeframe=timeframe,
                    count=config.HISTORY_LIMIT
                )
                if not candles or len(candles) == 0:
                    logging.warning(f"Данные для {symbol} отсутствуют")
                    raise ValueError(f"Данные для {symbol} отсутствуют")

                df = pd.DataFrame(candles)
                logging.debug(f"Получены данные: shape={df.shape}, columns={df.columns.tolist()}")
                df = df.rename(columns={
                    "time": "timestamp",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "tickVolume": "volume"
                })
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                df = df.sort_values("timestamp").tail(config.HISTORY_LIMIT)

                if df.empty or len(df) < 50:
                    logging.error(f"Недостаточно данных для {symbol}: {len(df)} строк")
                    raise ValueError(f"Недостаточно данных для {symbol}: {len(df)} строк")

                required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                if not all(col in df.columns for col in required_cols):
                    logging.error(f"Отсутствуют столбцы: {required_cols}, получено: {df.columns.tolist()}")
                    raise ValueError(f"Отсутствуют столбцы: {required_cols}, получено: {df.columns.tolist()}")

                logging.info(f"Успешно загружены данные для {symbol}: shape={df.shape}, columns={df.columns.tolist()}, timeframe={config.TIMEFRAME.get(symbol)}")
                return df[required_cols]

            except Exception as e:
                logging.error(f"Ошибка получения свечей на попытке {attempt + 1} для {symbol}: {e}")
                if attempt < 2:
                    logging.info(f"Повтор через 5 секунд")
                    await asyncio.sleep(5)
                else:
                    raise ValueError(f"Не удалось загрузить данные для {symbol} после 3 попыток: {e}")

    except Exception as e:
        logging.error(f"Общая ошибка загрузки данных для {symbol}: {e}")
        raise Exception(f"Ошибка загрузки данных для {symbol}: {e}")
    finally:
        # Безопасное отключение
        try:
            if account is not None:
                logging.info(f"Отключение аккаунта для {symbol}")
                await account.disconnect()
            if api is not None:
                logging.debug("Отключение MetaApi")
                await api.disconnect()
        except Exception as e:
            logging.warning(f"Ошибка при очистке ресурсов для {symbol}: {e}")

# Для синхронного вызова
def fetch_data_sync(symbol):
    logging.debug(f"Синхронный запрос данных для {symbol}")
    return asyncio.run(fetch_data(symbol))

def prepare_features(df, lookahead_days=4):
    logging.info("Подготовка признаков")
    try:
        df = df.copy()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logging.error(f"Отсутствуют необходимые столбцы: {required_cols}, получено: {df.columns.tolist()}")
            raise ValueError(f"Отсутствуют необходимые столбцы: {required_cols}, получено: {df.columns.tolist()}")

        df = df.reset_index(drop=True)
        logging.debug(f"Исходный DataFrame: shape={df.shape}, columns={df.columns.tolist()}")

        steps = lookahead_days
        if len(df) < steps + 50:
            logging.error(f"Недостаточно данных: {len(df)} строк, требуется минимум {steps + 50}")
            raise ValueError(f"Недостаточно данных: {len(df)} строк, требуется минимум {steps + 50}")

        df['target'] = df['close'].shift(-steps)
        if df['target'].isna().all():
            logging.error(f"Столбец 'target' содержит только NaN, недостаточно данных для сдвига на {steps}")
            raise ValueError(f"Столбец 'target' содержит только NaN, недостаточно данных для сдвига на {steps}")

        df = df.dropna(subset=['target', 'close'])
        if df.empty:
            logging.error("DataFrame пуст после удаления NaN в 'target' и 'close'")
            raise ValueError("DataFrame пуст после удаления NaN в 'target' и 'close'")

        df['direction'] = (df['target'] > df['close']).astype(int)

        df['ma_short'] = df['close'].rolling(window=10, min_periods=10).mean()
        df['ma_long'] = df['close'].rolling(window=50, min_periods=50).mean()
        df['volatility'] = df['high'] - df['low']
        df['momentum'] = df['close'].diff(5)

        df = df.dropna()
        if df.empty:
            logging.error("DataFrame пуст после удаления NaN для признаков")
            raise ValueError("DataFrame пуст после удаления NaN для признаков")

        features = ['open', 'high', 'low', 'close', 'volume', 'ma_short', 'ma_long', 'volatility', 'momentum']
        X = df[features]
        y = df['direction']

        logging.info(f"Признаки подготовлены: X.shape={X.shape}, y.shape={y.shape}")
        return X, y
    except Exception as e:
        logging.error(f"Ошибка подготовки признаков: {e}")
        raise Exception(f"Ошибка подготовки признаков: {e}")

def optimize_model_params(X, y):
    logging.info("Оптимизация параметров модели")
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

        logging.info(f"Оптимизация завершена: best_params={best_params}, best_score={best_score * 100:.2f}%")
        return grid_search.best_estimator_, best_score
    except Exception as e:
        logging.error(f"Ошибка оптимизации модели: {e}")
        raise Exception(f"Ошибка оптимизации модели: {e}")

def train_model(X, y):
    logging.info("Обучение модели")
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        logging.info(f"Точность модели: {accuracy * 100:.2f}%")

        if accuracy >= 0.8:
            logging.info("Модель прошла порог точности")
            return model

        logging.warning("Точность ниже 80%, запуск оптимизации...")
        optimized_model, _ = optimize_model_params(X, y)
        acc = accuracy_score(y_test, optimized_model.predict(X_test))
        logging.info(f"Точность оптимизированной модели: {acc * 100:.2f}%")

        if acc < 0.8:
            logging.error(f"Не удалось достичь точности 80%: {acc * 100:.2f}%")
            raise ValueError(f"Не удалось достичь точности 80%: {acc * 100:.2f}%")

        return optimized_model
    except Exception as e:
        logging.error(f"Ошибка обучения модели: {e}")
        raise Exception(f"Ошибка обучения модели: {e}")
