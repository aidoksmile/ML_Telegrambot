from flask import Flask
from predictor import generate_signals

app = Flask(__name__)

@app.route('/')
def home():
    signal = generate_signals()
    return f"Signal for EURUSD: {signal}"
