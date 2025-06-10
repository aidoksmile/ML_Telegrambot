# main.py

import time
import pandas as pd
from datetime import datetime
import requests
import config
import model
from strategy import MLStrategy
from backtesting import Backtest


def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage" 
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    requests.post(url, json=payload)


def run_backtest(df, model_obj):
    bt_df = df.copy()
    bt_df.reset_index(inplace=True)
    bt_df.rename(columns={'timestamp': 'index'}, inplace=True)
    bt_df.set_index('index', inplace=True)

    bt = Backtest(bt_df, MLStrategy, cash=10_000, commission=0.0002)
    bt._broker._model = model_obj
    bt._broker._features = bt_df[['open', 'high', 'low', 'close', 'volume', 'ma_short', 'ma_long', 'volatility', 'momentum']]
    result = bt.run()
    return result


def generate_signal(df, model_obj):
    last_row = df.iloc[-1]
    features = last_row[['open', 'high', 'low', 'close', 'volume', 'ma_short', 'ma_long', 'volatility', 'momentum']].values.reshape(1, -1)
    direction = model_obj.predict(features)[0]
    entry_price = last_row['close']
    volatility = last_row['volatility']
    stop_loss = entry_price - 2 * volatility
    take_profit = entry_price + 4 * volatility
    return direction, entry_price, stop_loss, take_profit


def process_symbol(symbol):
    print(f"[{datetime.now()}] Обработка пары: {symbol}")
    try:
        df = model.fetch_data(symbol)
    except Exception as e:
        error_msg = f"❌ Не удалось загрузить данные для {symbol}: {e}"
        print(error_msg)
        send_telegram_message(error_msg)
        return

    X, y = model.prepare_features(df, config.LOOKAHEAD_DAYS)
    try:
        model_obj = model.train_model(X, y)
    except Exception as e:
        print(f"❌ Не удалось обучить модель: {e}")
        send_telegram_message(f"❌ Не удалось обучить модель для {symbol}: {e}")
        return

    # Генерация сигнала
    direction, entry, sl, tp = generate_signal(df, model_obj)

    # Бэктестинг
    stats = run_backtest(df, model_obj)
    roi = stats["Return [%]"]
    win_rate = stats["Win Rate [%]"]

    # Отправка сигнала в Telegram
    signal_text = f"""
🔔 **Сигнал для {symbol}**
🕒 Время: {datetime.now()}
📊 Направление: {'BUY' if direction == 1 else 'SELL'}
💰 Цена входа: {entry:.2f}
📉 Stop Loss: {sl:.2f}
📈 Take Profit: {tp:.2f}
📊 ROI: {roi:.2f}%
🎯 Win Rate: {win_rate:.2f}%
    """
    send_telegram_message(signal_text)
    print(f"✅ Сигнал отправлен для {symbol}")


def main():
    print("[Бот запущен]")
    send_telegram_message("🟢 Бот запущен и готов к работе!")

    while True:
        try:
            for symbol in config.ASSETS:
                process_symbol(symbol)
            print("[Сон...] Следующее обновление через 24ч")
            time.sleep(config.UPDATE_INTERVAL)
        except Exception as e:
            error_msg = f"❌ Ошибка в основном цикле: {e}"
            print(error_msg)
            send_telegram_message(error_msg)
            time.sleep(60)


if __name__ == "__main__":
    main()
