# server.py

from flask import Flask
import threading
import main

app = Flask(__name__)

@app.route("/start")
def start_bot():
    thread = threading.Thread(target=main.main)
    thread.daemon = True
    thread.start()
    return "Бот запущен в фоне!", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
