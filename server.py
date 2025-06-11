import logging
import threading
import time
from main import process_assets, send_telegram_message
import config

logging.basicConfig(
    filename="bot.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def periodic_processing():
    logging.info("Запуск периодической обработки активов")
    while True:
        try:
            process_assets()
            logging.info(f"Ожидание {config.UPDATE_INTERVAL} секунд до следующей обработки")
        except Exception as e:
            logging.error(f"Ошибка в периодической обработке: {e}")
            send_telegram_message(f"❌ Ошибка в периодической обработке: {e}")
        time.sleep(config.UPDATE_INTERVAL)

if __name__ == "__main__":
    logging.info("Запуск бота")
    threading.Thread(target=periodic_processing, daemon=True).start()
    # Бесконечный цикл для работы на Render
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Остановка бота")
