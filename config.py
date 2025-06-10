# config.py

TELEGRAM_BOT_TOKEN = "8132037717:AAHK04i4cHkgqH-Qyiy7OHs73xbXmvqnlpc"  # Получите через @BotFather
TELEGRAM_CHAT_ID = "106240757"      # Получите через getUpdates

ASSETS = ["XAU/USD", "EUR/USD"]  # Список символов для итерации

# Маппинг символов к тикерам Yahoo Finance
TICKER_MAP = {
    "XAU/USD": "GC=F",
    "EUR/USD": "EURUSD=X"
}

TIMEFRAME = "1d"
LOOKAHEAD_DAYS = 4
HISTORY_LIMIT = 500
UPDATE_INTERVAL = 3600  # Уменьшено до 1 часа для теста
