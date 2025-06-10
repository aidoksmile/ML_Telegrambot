# config.py

TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"  # Получите через @BotFather
TELEGRAM_CHAT_ID = "TELEGRAM_CHAT_ID"      # Получите через getUpdates

CAPITAL_API_KEY = "CAPITAL_API_KEY"      # Ваш API-ключ от Capital.com
CAPITAL_API_PASSWORD = "CAPITAL_API_PASSWORD"    # Пароль для API-ключа (если задан)
CAPITAL_EMAIL = "CAPITAL_EMAIL"                  # Ваш email на Capital.com
CAPITAL_PASSWORD = "CAPITAL_PASSWORD"            # Пароль от аккаунта Capital.com

ASSETS = ["XAU/USD", "EUR/USD"]
TICKER_MAP = {
    "XAU/USD": "GOLD",      # Epic для золота (уточните в GET /markets)
    "EUR/USD": "EURUSD"     # Epic для валютной пары
}
TIMEFRAME = {
    "XAU/USD": "MINUTE_15",
    "EUR/USD": "MINUTE_15"
}
LOOKAHEAD_DAYS = 4        # 4 * 15 минут = 1 час
HISTORY_LIMIT = 200       # ~50 часов данных
UPDATE_INTERVAL = 900     # 15 минут
BASE_URL = "https://demo-api-capital.backend-capital.com"  # Для демо-аккаунта
