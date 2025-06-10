import logging
import model
import config
import requests
import time
from datetime import datetime

logging.basicConfig(
    filename="bot.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": message
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logging.info(f"Сообщение отправлено в Telegram: {message}")
    except Exception as e:
        logging.error(f"Ошибка отправки сообщения в Telegram: {e}")

def process_assets():
    logging.info("Начало обработки активов")
    for symbol in config.ASSETS:
        try:
            logging.info(f"Обработка актива: {symbol}")
            df = model.fetch_data(symbol)
            X, y = model.prepare_features(df, config.LOOKAHEAD_DAYS)
            params = model.optimize_model_params(X, y)
            model_instance = model.train_model(X, y, params)
            
            latest_data = X.iloc[-1].values.reshape(1, -1)
            prediction = model_instance.predict(latest_data)[0]
            probability = model_instance.predict_proba(latest_data)[0][prediction]
            
            direction = "вверх" if prediction == 1 else "вниз"
            message = (
                f"📊 Прогноз для {symbol} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):\n"
                f"Направление: {direction}\n"
                f"Вероятность: {probability:.2%}\n"
                f"Текущая цена: {df['close'].iloc[-1]:.4f}"
            )
            send_telegram_message(message)
            logging.info(f"Прогноз для {symbol} успешно обработан")
        except Exception as e:
            error_message = f"❌ Ошибка обработки {symbol}: {e}"
            logging.error(error_message)
            send_telegram_message(error_message)

if __name__ == "__main__":
    process_assets()
