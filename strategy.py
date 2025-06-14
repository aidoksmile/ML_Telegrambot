import logging
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from server import send_message

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)

# Конфигурация
ASSETS = ["XAUUSD", "EURUSD", "USDJPY"]
TICKER_MAP = {"XAUUSD": "GC=F", "EURUSD": "EURUSD=X", "USDJPY": "JPY=X"}
HISTORY_LIMIT = 200

def fetch_data(symbol):
    """Загружает данные с Yahoo Finance с проверкой."""
    ticker = TICKER_MAP[symbol]
    logging.info(f"Попытка загрузки данных для {symbol} (тикер: {ticker})...")
    try:
        df = yf.download(ticker, period="60d", interval="15m")
        if df.empty:
            logging.error(f"Данные для {symbol} ({ticker}) отсутствуют или недоступны")
            raise ValueError(f"Данные для {symbol} отсутствуют или недоступны")
        logging.info(f"Успешно загружено {len(df)} записей для {symbol}")
        df = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
        })
        df = df.tail(HISTORY_LIMIT)
        logging.debug(f"Данные для {symbol} обрезаны до {HISTORY_LIMIT} записей")
        return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных для {symbol}: {e}")
        raise

def prepare_features(df):
    """Подготавливает признаки для модели."""
    df = df.copy()
    df['target'] = df['close'].shift(-4)
    df = df.dropna(subset=['target'])
    df['direction'] = (df['target'] > df['close']).astype(int)
    df['ma_short'] = df['close'].rolling(window=10).mean()
    df['ma_long'] = df['close'].rolling(window=50).mean()
    df['volatility'] = df['high'] - df['low']
    df['momentum'] = df['close'].diff(5)
    df = df.dropna()
    features = ['open', 'high', 'low', 'close', 'volume', 'ma_short', 'ma_long', 'volatility', 'momentum']
    logging.debug(f"Подготовлено {len(df)} записей с признаками для модели")
    return df[features], df['direction']

def train_model(X, y):
    """Обучает модель."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    logging.info(f"Точность модели для {X.name}: {accuracy * 100:.2f}%")
    return model

def process_asset(symbol):
    """Обрабатывает актив и генерирует сигнал."""
    try:
        df = fetch_data(symbol)
        X, y = prepare_features(df)
        X.name = symbol
        model = train_model(X, y)
        latest = X.iloc[-1].values.reshape(1, -1)
        signal = "Покупка" if model.predict(latest)[0] == 1 else "Продажа"
        logging.info(f"Сгенерирован сигнал для {symbol}: {signal}")
        send_message(f"📈 Сигнал для {symbol}: {signal}")
    except Exception as e:
        logging.error(f"Ошибка обработки {symbol}: {e}")
        send_message(f"❌ Ошибка обработки {symbol}: {e}")

def generate_signals():
    """Генерирует сигналы для всех активов."""
    logging.info("Начало генерации сигналов")
    for symbol in ASSETS:
        process_asset(symbol)
    logging.info("Генерация сигналов завершена")
