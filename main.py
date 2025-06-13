import asyncio
import os
import logging
from telegram.ext import Application, CommandHandler
import aiocron
from server import send_message, start_server
from strategy import generate_signals

# Настройка логирования
logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Конфигурация
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logging.error("Отсутствуют TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID")
    raise ValueError("Отсутствуют необходимые переменные окружения")

async def signal_command(update, context):
    """Обработчик команды /signal."""
    await update.message.reply_text("⏳ Генерация сигналов...")
    await generate_signals()

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

        # Запуск HTTP-сервера
        await start_server()
        logging.info("HTTP-сервер и Telegram-бот запущены")

        # Планировщик: 01:00 UTC (6:00 +05)
        aiocron.crontab('0 1 * * *', func=generate_signals, start=True)
        logging.info("Планировщик запущен")

        # Бесконечный цикл
        while True:
            await asyncio.sleep(3600)

    except Exception as e:
        logging.critical(f"Критическая ошибка: {e}")
        await send_message(f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(main())
