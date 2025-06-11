import os
import logging

# Настройка логирования
logging.basicConfig(
    filename="bot.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

# Загрузка переменных окружения
def load_env_variable(var_name):
    value = os.getenv(var_name)
    if value is None:
        logging.error(f"Переменная окружения {var_name} не задана")
        raise ValueError(f"Переменная окружения {var_name} не задана")
    logging.debug(f"Переменная {var_name} загружена")
    return value

# Обязательные переменные
try:
    TELEGRAM_BOT_TOKEN = load_env_variable("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = load_env_variable("TELEGRAM_CHAT_ID")
    METAAPI_TOKEN = load_env_variable("METAAPI_TOKEN")
    MT4_ACCOUNT_ID = load_env_variable("MT4_ACCOUNT_ID")
except ValueError as e:
    logging.critical(f"Ошибка конфигурации: {e}")
    raise

# Конфигурация активов
ASSETS = ["XAUUSD", "EURUSD"]
TICKER_MAP = {
    "XAUUSD": "XAUUSD",
    "EURUSD": "EURUSD"
}

# Таймфреймы
TIMEFRAME = {
    "XAUUSD": "15m",
    "EURUSD": "15m"
}

# Настройки данных
HISTORY_LIMIT = 200  # Количество свечей для загрузки
UPDATE_INTERVAL = 900  # Интервал обновления в секундах (15 минут)

logging.info("Конфигурация успешно загружена")
