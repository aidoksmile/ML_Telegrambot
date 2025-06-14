from telegram import Bot
import os

TELEGRAM_TOKEN = os.environ.get("8132037717:AAHK04i4cHkgqH-Qyiy7OHs73xbXmvqnlp")
CHAT_ID = os.environ.get("106240757")
bot = Bot(token=TELEGRAM_TOKEN)

def send_telegram_message(text):
    if TELEGRAM_TOKEN and CHAT_ID:
        bot.send_message(chat_id=CHAT_ID, text=text)
    else:
        print("Telegram credentials not set.")
