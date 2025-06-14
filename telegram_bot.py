from telegram import Bot

TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

def send_signal(signal):
    text = f'''
📈 Пара: {signal["pair"]}
📉 Направление: {signal["direction"]}
💰 Текущая цена: {signal["entry"]:.2f}
🛑 SL: {signal["sl"]:.2f}
🎯 TP: {signal["tp"]:.2f}
⏱ Прогноз на 1–4 дня
'''
    Bot(token=TOKEN).send_message(chat_id=CHAT_ID, text=text)