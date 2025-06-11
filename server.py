import logging
import threading
import time
from main import process_assets, send_telegram_message
import config
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(
    filename="bot.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Минимальный HTTP-обработчик для Render
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bot is running")

async def periodic_processing():
    logging.info("Запуск периодической обработки активов")
    while True:
        try:
            await process_assets()
            logging.info(f"Ожидание {config.UPDATE_INTERVAL} секунд до следующей обработки")
        except Exception as e:
            logging.error(f"Ошибка в периодической обработке: {e}")
            send_telegram_message(f"❌ Ошибка в периодической обработке: {e}")
        time.sleep(config.UPDATE_INTERVAL)

def run_http_server():
    # Запуск HTTP-сервера на порту 10000
    server_address = ('', 10000)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    logging.info("Запуск HTTP-сервера на порту 10000")
    httpd.serve_forever()

if __name__ == "__main__":
    logging.info("Запуск бота")
    # Запуск асинхронной обработки в отдельном потоке
    def run_async_loop():
        asyncio.run(periodic_processing())
    
    threading.Thread(target=run_async_loop, daemon=True).start()
    # Запуск HTTP-сервера в главном потоке
    try:
        run_http_server()
    except KeyboardInterrupt:
        logging.info("Остановка бота")
