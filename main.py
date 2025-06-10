import time
import pandas as pd
from datetime import datetime
import requests
import config
import model
from strategy import MLStrategy
from backtesting import Backtest

print("[INFO] main.py загружен")

def send_telegram_message(text):
    try:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"[INFO] Сообщение отправлено в Telegram: {text[:50]}...")
        return response
    except Exception as e:
        print(f"[ERROR] Ошибка отправки в Telegram: {e}")
        return None

def run_backtest(df, model_obj):
    try:
        bt_df = df.copy()
        bt_df.reset_index(inplace=True)
        bt_df.rename(columns={'timestamp': 'index'}, inplace=True)
        bt_df.set_index('index', inplace=True)

        bt = Backtest(bt_df, MLStrategy, cash=10_000, commission=0.0002)
        bt._broker._model = model_obj
        bt._broker._features = bt_df[['open', 'high', 'low', 'close', 'volume', 'ma_short', 'ma_long', 'volatility', 'momentum']]
        result = bt.run()
        print(f"[INFO] Бэктест завершен, ROI: {result['Return [%]']:.2f}%")
        return result
    except Exception as e:
        print(f"[ERROR] Ошибка в бэктестинге: {e}")
        return None

def generate_signal(df, model_obj):
    try:
        last_row = df.iloc[-1]
        features = last_row[['open', 'high', 'low', 'close', 'volume', 'ma_short', 'ma_long', 'volatility', 'momentum']].values.reshape(1, -1)
        direction = model_obj.predict(features)[0]
        entry_price = last_row['close']
        volatility = last_row['volatility']
        stop_loss = entry_price - 2 * volatility
        take_profit = entry_price + 4 * volatility
        print(f"[INFO] Сигнал сгенерирован: {direction}, Entry: {entry_price:.2f}")
        return direction, entry_price, stop_loss, take_profit
    except Exception as e:
        print(f"[ERROR] Ошибка при генерации сигнала: {e}")
        return None, None, None, None

def process_symbol(symbol):
    print(f"[{datetime.now()}] Начало обработки пары: {symbol}")
    try:
        df = model.fetch_data(symbol)
        print(f"[DEBUG] Данные для {symbol} загружены, размер: {df.shape}")
        if df.empty:
            error_msg = f"❌ Данные для {symbol} пусты"
            print(error_msg)
            send_telegram_message(error_msg)
            return
    except Exception as e:
        error_msg = f"❌ Не удалось загрузить данные для {symbol}: {e}"
        print(error_msg)
        send_telegram_message(error_msg)
        return

    try:
        X, y = model.prepare_features(df, config.LOOKAHEAD_DAYS)
        print(f"[DEBUG] Features для {symbol}: X.shape={X.shape}, y.shape={y.shape}")
        if X.empty or y.empty:
            error_msg = f"❌ Недостаточно данных для обучения модели {symbol}"
            print(error_msg)
            send_telegram_message(error_msg)
            return
    except Exception as e:
        error_msg = f"❌ Не удалось подготовить данные для {symbol}: {e}"
        print(error_msg)
        send_telegram_message(error_msg)
        return

    try:
        model_obj = model.train_model(X, y)
        print(f"[INFO] Модель для {symbol} обучена")
    except Exception as e:
        error_msg = f"❌ Не удалось обучить модель для {symbol}: {e}"
        print(error_msg)
        send_telegram_message(error_msg)
        return

    direction, entry, sl, tp = generate_signal(df, model_obj)
    if direction is None:
        error_msg = f"❌ Не удалось сгенерировать сигнал для {symbol}"
        print(error_msg)
        send_telegram_message(error_msg)
        return

    stats = run_backtest(df, model_obj)
    if stats is None:
        error_msg = f"❌ Не удалось выполнить бэктест для {symbol}"
        print(error_msg)
        send_telegram_message(error_msg)
        return

    roi = stats["Return [%]"]
    win_rate = stats["Win Rate [%]"]

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
    print(f"[CONFIG] ASSETS: {config.ASSETS}, UPDATE_INTERVAL: {config.UPDATE_INTERVAL}")
    print("[START] Запуск бота...")
    send_telegram_message("🟢 Бот запущен и готов к работе!")
    print("[Бот запущен]")

    while True:
        try:
            for symbol in config.ASSETS:
                process_symbol(symbol)
            print(f"[Сон...] Следующее обновление через {config.UPDATE_INTERVAL} секунд")
            time.sleep(config.UPDATE_INTERVAL)
        except Exception as e:
            error_msg = f"⚠️ Ошибка в основном цикле: {e}"
            print(error_msg)
            send_telegram_message(error_msg)
            time.sleep(60)

if __name__ == "__main__":
    main()
