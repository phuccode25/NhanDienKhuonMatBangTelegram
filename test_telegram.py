from notifier import TelegramNotifier

TELEGRAM_BOT_TOKEN = "7770244537:AAHCaPGU-_5E7D1EZancGITXqt_OZ6idtBI"
TELEGRAM_CHAT_ID = "6269445809"

noti = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
noti.send_text("Test cảnh báo từ intruder_system")
