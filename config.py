TELEGRAM_BOT_TOKEN = "8132037717:AAHK04i4cHkgqH-Qyiy7OHs73xbXmvqnlpc"  # Получите через @BotFather
TELEGRAM_CHAT_ID = "106240757"      # Получите через getUpdates

MT4_LOGIN = "420493"                    # Логин вашего MT4-аккаунта
MT4_PASSWORD = "MT4_PASSWORD"              # Пароль вашего MT4-аккаунта
MT4_SERVER = "Capitol.com-Demo"                  # Сервер брокера (например, "MetaQuotes-Demo")

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
