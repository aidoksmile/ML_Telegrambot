from telegram import Bot

TOKEN = "8132037717:AAHK04i4cHkgqH-Qyiy7OHs73xbXmvqnlpc"
CHAT_ID = "106240757"

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
