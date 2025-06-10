import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
from main import process_assets, send_telegram_message
import config

logging.basicConfig(
    filename="bot.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        logging.info("Получен HTTP-запрос")
        try:
            threading.Thread(target=process_assets, daemon=True).start()
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Request received, processing assets in background")
        except Exception as e:
            logging.error(f"Ошибка обработки запроса: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Error: {e}".encode())

def run_server():
    server_address = ("", 10000)
    httpd = HTTPServer(server_address, RequestHandler)
    logging.info("Запуск HTTP-сервера на порту 10000")
    httpd.serve_forever()

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
    try:
        run_server()
    except Exception as e:
        logging.error(f"Ошибка сервера: {e}")
        send_telegram_message(f"❌ Ошибка сервера: {e}")
