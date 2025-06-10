from flask import Flask
import threading
import main
import logging

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('bot.log')]
)

bot_running = False
bot_thread = None

@app.route("/")
def start_bot():
    global bot_running, bot_thread
    app.logger.info("Получен запрос к /")

    if bot_running:
        app.logger.info("Бот уже запущен")
        return "🚀 Бот уже запущен!", 200

    def run_bot():
        try:
            app.logger.info("Запуск main.main()")
            main.main()
        except Exception as e:
            app.logger.error(f"Ошибка в основном цикле: {e}")

    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()
    bot_running = True
    app.logger.info("Бот запущен в фоне")
    return "🚀 Бот запущен в фоне!", 200

@app.route("/status")
def check_status():
    if bot_running and bot_thread.is_alive():
        return "🚀 Бот работает!", 200
    else:
        return "⚠️ Бот не запущен или завершился!", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
