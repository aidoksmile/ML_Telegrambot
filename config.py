TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"  # Получите через @BotFather
TELEGRAM_CHAT_ID = "TELEGRAM_CHAT_ID"      # Получите через getUpdates

MT4_LOGIN = "MT4_LOGIN"                    # Логин вашего MT4-аккаунта
MT4_PASSWORD = "MT4_PASSWORD"              # Пароль вашего MT4-аккаунта
MT4_SERVER = "MT4_SERVER"                  # Сервер брокера (например, "MetaQuotes-Demo")

ASSETS = ["XAUUSD", "EURUSD"]
TICKER_MAP = {
    "XAUUSD": "XAUUSD",
    "EURUSD": "EURUSD"
}
TIMEFRAME = {
    "XAUUSD": "15m",
    "EURUSD": "15m"
}
LOOKAHEAD_DAYS = 4        # 4 * 15 минут = 1 час
HISTORY_LIMIT = 200       # ~50 часов данных
UPDATE_INTERVAL = 900     # 15 минут
