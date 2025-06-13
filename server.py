import os
import logging
import aiohttp
from aiohttp import web

# Настройка логирования
logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Конфигурация
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logging.error("Отсутствуют TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID")
    raise ValueError("Отсутствуют необходимые переменные окружения")

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

async def health_check(request):
    """Проверка работоспособности для Render."""
    return web.Response(text="Bot is running")

async def start_server():
    """Запускает HTTP-сервер."""
    app = web.Application()
    app.router.add_get('/health', health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 10000)
    await site.start()
    logging.info("HTTP-сервер запущен")
