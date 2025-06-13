import asyncio
import os
import logging
from telegram.ext import Application, CommandHandler
import aiocron
from server import send_message, start_server
from strategy import generate_signals

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Вывод только в консоль
)

# Конфигурация
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logging.error("Отсутствуют TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID")
    raise ValueError("Отсутствуют необходимые переменные окружения")

async def signal_command(update, context):
    """Обработчик команды /signal."""
    logging.debug("Получена команда /signal")
    await update.message.reply_text("⏳ Генерация сигналов...")
    await generate_signals()

async def run_bot():
    """Запуск Telegram-бота."""
    try:
        logging.debug("Инициализация Telegram-бота")
        app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        app.add_handler(CommandHandler("signal", signal_command))
        await app.initialize()
        logging.info("Telegram-бот инициализирован")
        
        logging.debug("Отправка тестового сообщения")
        await send_message("✅ Бот запущен")

        logging.debug("Запуск polling")
        await app.run_polling(allowed_updates=[])  # Блокирует выполнение
    except Exception as e:
        logging.critical(f"Ошибка бота: {e}")
        await send_message(f"❌ Критическая ошибка бота: {e}")

async def run_server():
    """Запуск HTTP-сервера в фоновом режиме."""
    try:
        logging.debug("Запуск HTTP-сервера")
        await start_server()
        logging.info("HTTP-сервер запущен")
    except Exception as e:
        logging.critical(f"Ошибка сервера: {e}")
        await send_message(f"❌ Критическая ошибка сервера: {e}")

async def run_scheduler():
    """Настройка планировщика."""
    logging.debug("Настройка планировщика")
    aiocron.crontab('0 1 * * *', func=generate_signals, start=True)
    logging.info("Планировщик запущен")

async def main():
    """Главная функция."""
    try:
        # Запуск сервера и планировщика в фоновом режиме
        server_task = asyncio.create_task(run_server())
        scheduler_task = asyncio.create_task(run_scheduler())

        # Запуск бота (блокирует выполнение)
        await run_bot()

        # Ожидание завершения задач (не будет достигнуто из-за run_polling)
        await asyncio.gather(server_task, scheduler_task)
    except Exception as e:
        logging.critical(f"Критическая ошибка: {e}")
        await send_message(f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(main())
