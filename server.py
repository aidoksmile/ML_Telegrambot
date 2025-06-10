# server.py

from flask import Flask
import threading
import main
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/")
def start_bot():
    def run_bot():
        try:
            main.main()
        except Exception as e:
            app.logger.error(f"Ошибка в основном цикле: {e}")

    thread = threading.Thread(target=run_bot)
    thread.daemon = True
    thread.start()
    return "🚀 Бот запущен в фоне!", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
