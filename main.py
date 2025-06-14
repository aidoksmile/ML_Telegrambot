import os
import logging
import time
import schedule
from telegram.ext import Updater, CommandHandler
from server import send_message, start_server
from strategy import generate_signals

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)

# Конфигурация
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logging.error("Отсутствуют TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID")
    raise ValueError("Отсутствуют необходимые переменные окружения")
if ':' not in TELEGRAM_BOT_TOKEN or len(TELEGRAM_BOT_TOKEN) < 10:
    logging.error("Некорректный формат TELEGRAM_BOT_TOKEN")
    raise ValueError("Некорректный формат TELEGRAM_BOT_TOKEN")

def signal_command(update, context):
    """Обработчик команды /signal."""
    logging.debug("Получена команда /signal")
    update.message.reply_text("⏳ Генерация сигналов...")
    generate_signals()

def main():
    """Главная функция."""
    updater = None
    try:
        # Проверка токена
        logging.info(f"Проверка токена: {TELEGRAM_BOT_TOKEN[:10]}...")  # Обрезаем для безопасности

        # Запуск Telegram-бота
        logging.info("Инициализация Telegram-бота...")
        updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
        updater.dispatcher.add_handler(CommandHandler("signal", signal_command))
        updater.start_polling()
        logging.info("Telegram-бот запущен")

        # Тестовое сообщение
        logging.info("Отправка тестового сообщения...")
        send_message("✅ Бот запущен")

        # Запуск HTTP-сервера
        logging.info("Запуск HTTP-сервера...")
        start_server()
        logging.info("HTTP-сервер запущен")

        # Планировщик: 01:00 UTC (6:00 +05)
        logging.info("Запуск планировщика...")
        schedule.every().day.at("01:00").do(generate_signals)
        logging.info("Планировщик запущен")

        # Бесконечный цикл
        while True:
            schedule.run_pending()
            time.sleep(60)

    except Exception as e:
        logging.critical(f"Критическая ошибка бота: {e}", exc_info=True)
        send_message(f"❌ Критическая ошибка бота: {e}")
    finally:
        if updater:
            updater.stop()

if __name__ == "__main__":
    main()
