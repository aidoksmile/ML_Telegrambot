import os
import logging
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler

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

def send_message(message):
    """Отправляет сообщение в Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.error(f"Ошибка Telegram: {response.text}")
        else:
            logging.info(f"Сообщение отправлено: {message}")
    except Exception as e:
        logging.error(f"Ошибка отправки сообщения: {e}")

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

def start_server():
    """Запускает HTTP-сервер."""
    server_address = ('0.0.0.0', 10000)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    logging.info("HTTP-сервер запущен на порту 10000")
    httpd.serve_forever()
