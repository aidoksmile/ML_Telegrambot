import logging
import threading
import time
from main import process_assets, send_telegram_message
import config
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler

# Настройка логирования
logging.basicConfig(
    filename="bot.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"  # Добавляем, чтобы не перезаписывать лог
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
        time.sleep(config.UPDATE_INTERVAL)

def run_async_loop():
    try:
        logging.info("Запуск асинхронного цикла")
        asyncio.run(periodic_processing())
    except Exception as e:
        logging.error(f"Ошибка в асинхронном цикле: {str(e)}")

def run_http_server():
    server_address = ('', 10000)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    logging.info("Запуск HTTP-сервера на порту 10000")
    try:
        httpd.serve_forever()
    except Exception as e:
        logging.error(f"Ошибка HTTP-сервера: {str(e)}")

if __name__ == "__main__":
    logging.info("Запуск бота")
    # Запуск асинхронной обработки в отдельном потоке
    threading.Thread(target=run_async_loop, daemon=True).start()
    # Запуск HTTP-сервера в главном потоке
    try:
        run_http_server()
    except KeyboardInterrupt:
        logging.info("Остановка бота")
