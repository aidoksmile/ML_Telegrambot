import joblib
import yfinance as yf
from utils import add_indicators, calculate_sl_tp, get_target

def generate_signals():
    model = joblib.load("model.pkl")
    df = yf.download("XAUUSD=X", period="7d", interval="15m")
    df = add_indicators(df)
    df["target"] = get_target(df)
    X = df.drop(columns=["target"]).dropna()
    if X.empty:
        return None
    prediction = model.predict(X.tail(1))[0]
    current_price = df["Close"].iloc[-1]
    sl, tp = calculate_sl_tp(current_price, prediction)
    return {
        "pair": "XAUUSD",
        "direction": "BUY" if prediction == 1 else "SELL",
        "entry": current_price,
        "sl": sl,
        "tp": tp
    }