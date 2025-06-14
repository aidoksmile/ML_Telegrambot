from flask import Flask, jsonify
from predictor import generate_signals

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    try:
        signal = generate_signals()
        return jsonify(signal)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
