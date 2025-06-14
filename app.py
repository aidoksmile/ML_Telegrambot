from flask import Flask
from telegram_bot import send_signal
from predictor import generate_signals

app = Flask(__name__)

@app.route('/')
def home():
    signal = generate_signals()
    if signal:
        send_signal(signal)
        return f"Signal sent: {signal}"
    return "No signal"

if __name__ == '__main__':
    app.run()