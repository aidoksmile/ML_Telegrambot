import asyncio
import os
import logging
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import aiohttp
from aiohttp import web
import aiocron
from telegram import Bot
from telegram.ext import Application, CommandHandler

# Настройка логирования
logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Конфигурация
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logging.error("Отсутствуют TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID")
    raise ValueError("Отсутствуют необходимые переменные окружения")

ASSETS = ["XAUUSD", "EURUSD", "USDJPY"]
TICKER_MAP = {"XAUUSD": "GC=F", "EURUSD": "EURUSD=X", "USDJPY": "JPY=X"}
HISTORY_LIMIT = 200

async def send_message(message):
    """Отправляет сообщение в Telegram."""
    async with aiohttp.ClientSession() as session:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                logging.error(f"Ошибка Telegram: {await response.text()}")
            else:
                logging.info(f"Сообщение отправлено: {message}")

def fetch_data(symbol):
    """Загружает данные с Yahoo Finance."""
    ticker = TICKER_MAP[symbol]
    df = yf.download(ticker, period="60d", interval="15m")
    if df.empty:
        logging.error(f"Данные для {symbol} отсутствуют")
        raise ValueError(f"Данные для {symbol} отсутствуют")
    df = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    })
    df = df.tail(HISTORY_LIMIT)
    return df

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
    return df[features], df['direction']

def train_model(X, y):
    """Обучает модель."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    logging.info(f"Точность модели для {X.name}: {accuracy * 100:.2f}%")
    return model

async def process_asset(symbol):
    """Обрабатывает актив и генерирует сигнал."""
    try:
        df = fetch_data(symbol)
        X, y = prepare_features(df)
        X.name = symbol
        model = train_model(X, y)
        latest = X.iloc[-1].values.reshape(1, -1)
        signal = "Покупка" if model.predict(latest)[0] == 1 else "Продажа"
        await send_message(f"📈 Сигнал для {symbol}: {signal}")
    except Exception as e:
        logging.error(f"Ошибка обработки {symbol}: {e}")
        await send_message(f"❌ Ошибка обработки {symbol}: {e}")

async def generate_signals():
    """Генерирует сигналы для всех активов."""
    logging.info("Генерация сигналов")
    for symbol in ASSETS:
        await process_asset(symbol)

async def signal_command(update, context):
    """Обработчик команды /signal."""
    await update.message.reply_text("⏳ Генерация сигналов...")
    await generate_signals()

async def health_check(request):
    """Проверка работоспособности для Render."""
    return web.Response(text="Bot is running")

async def main():
    """Главная функция."""
    try:
        # Запуск Telegram-бота
        app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        app.add_handler(CommandHandler("signal", signal_command))
        await app.initialize()
        await app.start()
        logging.info("Telegram-бот запущен")

        # Тестовое сообщение
        await send_message("✅ Бот запущен")

        # Планировщик: 01:00 UTC (6:00 +05)
        aiocron.crontab('0 1 * * *', func=generate_signals, start=True)
        logging.info("Планировщик запущен")

        # HTTP-сервер для Render
        web_app = web.Application()
        web_app.router.add_get('/health', health_check)
        runner = web.AppRunner(web_app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 10000)
        await site.start()
        logging.info("HTTP-сервер запущен")

        # Бесконечный цикл
        while True:
            await asyncio.sleep(3600)

    except Exception as e:
        logging.critical(f"Критическая ошибка: {e}")
        await send_message(f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(main())
