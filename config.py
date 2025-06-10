# config.py

TELEGRAM_BOT_TOKEN = "8132037717:AAHK04i4cHkgqH-Qyiy7OHs73xbXmvqnlpc"  # Получите через @BotFather
TELEGRAM_CHAT_ID = "106240757"      # Получите через getUpdates

ALPHA_VANTAGE_API_KEY = "WEGYRA74P069BV6J"    # Ваш API-ключ Alpha Vantage

ASSETS = ["XAU/USD", "EUR/USD"]
TICKER_MAP = {
    "XAU/USD": "XAUUSD",  # Золото
    "EUR/USD": "EURUSD"   # Валютная пара
}

# Интервалы: 'daily' для XAU/USD, '15min' для EUR/USD
TIMEFRAME = {
    "XAU/USD": "daily",
    "EUR/USD": "15min"
}
LOOKAHEAD_DAYS = 1        # Уменьшено для тестирования
HISTORY_LIMIT = 200       # Количество записей (для 15min это ~200 * 15 минут)
UPDATE_INTERVAL = 900     # 15 минут для тестирования
