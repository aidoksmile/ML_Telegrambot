import logging
import time
from main import process_assets, send_telegram_message
import config
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler

# Настройка логирования в файл и консоль
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log", mode="a"),
        logging.StreamHandler()  # Вывод в консоль
    ]
)

# Минимальный HTTP-обработчик для Render
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bot is running")
        logging.info("Получен GET-запрос на /")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        logging.info("Получен HEAD-запрос на /")

async def periodic_processing():
    logging.info("Запуск периодической обработки активов")
    while True:
        try:
            logging.debug("Начало цикла обработки активов")
            await process_assets()
            logging.info(f"Обработка завершена, ожидание {config.UPDATE_INTERVAL} секунд")
        except Exception as e:
            error_msg = f"Ошибка в периодической обработке: {str(e)}"
            logging.error(error_msg)
            try:
                send_telegram_message(f"❌ {error_msg}")
            except Exception as telegram_error:
                logging.error(f"Ошибка отправки в Telegram: {telegram_error}")
        await asyncio.sleep(config.UPDATE_INTERVAL)

async def main():
    logging.info("Запуск бота")
    # Отправка тестового сообщения в Telegram при старте
    try:
        send_telegram_message("✅ Бот запущен на Render")
        logging.info("Тестовое сообщение отправлено в Telegram")
    except Exception as e:
        logging.error(f"Ошибка отправки тестового сообщения в Telegram: {str(e)}")

    # Запуск HTTP-сервера в отдельном потоке
    def run_http_server():
        server_address = ('', 10000)
        httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
        logging.info("Запуск HTTP-сервера на порту 10000")
        try:
            httpd.serve_forever()
        except Exception as e:
            logging.error(f"Ошибка HTTP-сервера: {str(e)}")

    import threading
    threading.Thread(target=run_http_server, daemon=True).start()

    # Запуск периодической обработки
    await periodic_processing()

if __name__ == "__main__":
    asyncio.run(main())
