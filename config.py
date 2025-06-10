# config.py

TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "CHAT_ID"

ASSETS = {
    "XAU/USD": "GC=F",       # Фьючерс на золото
    "EUR/USD": "EURUSD=X"    # EUR/USD через Yahoo Finance
}

TIMEFRAME = "1d"            # yfinance не поддерживает M15 напрямую
LOOKAHEAD_DAYS = 4         # Прогноз на 4 дня вперёд
HISTORY_LIMIT = 500        # Количество дней истории
UPDATE_INTERVAL = 86400    # Обновление раз в день
