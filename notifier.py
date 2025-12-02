import os
import requests


class TelegramNotifier:
    """
    Gửi thông báo qua Telegram Bot:
    - send_text: gửi tin nhắn chữ
    - send_photo: gửi ảnh kèm caption
    - send_alert: tiện dùng cho cảnh báo (có thể kèm ảnh)
    """

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_text(self, text: str):
        """Gửi tin nhắn text đơn giản."""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {"chat_id": self.chat_id, "text": text}
            resp = requests.post(url, data=data, timeout=5)
            if not resp.ok:
                print("[TelegramNotifier] send_text lỗi:", resp.status_code, resp.text)
        except Exception as e:
            print("[TelegramNotifier] Lỗi gửi text:", e)

    def send_photo(self, photo_path: str, caption: str = ""):
        """Gửi ảnh kèm caption."""
        if not os.path.exists(photo_path):
            print("[TelegramNotifier] Không tìm thấy file ảnh:", photo_path)
            return

        try:
            url = f"{self.base_url}/sendPhoto"
            with open(photo_path, "rb") as f:
                files = {"photo": f}
                data = {"chat_id": self.chat_id, "caption": caption}
                resp = requests.post(url, data=data, files=files, timeout=10)
                if not resp.ok:
                    print("[TelegramNotifier] send_photo lỗi:", resp.status_code, resp.text)
        except Exception as e:
            print("[TelegramNotifier] Lỗi gửi ảnh:", e)

    def send_alert(self, message: str, image_path: str | None = None):
        """
        Hàm tiện dụng:
        - nếu có image_path -> gửi ảnh kèm message
        - nếu không -> gửi mỗi text
        """
        if image_path:
            self.send_photo(image_path, caption=message)
        else:
            self.send_text(message)
