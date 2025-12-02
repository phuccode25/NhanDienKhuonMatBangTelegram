import os
import time

import cv2


class AlertManager:
    """
    Xử lý phần cảnh báo:
    - Vẽ banner đỏ trên khung hình.
    - Lưu ảnh khi phát hiện người, nhưng giãn cách thời gian cho đỡ spam.
    """

    def __init__(self, output_dir: str = "alerts", min_interval: float = 3.0):
        """
        output_dir: thư mục lưu ảnh cảnh báo.
        min_interval: tối thiểu bao nhiêu giây mới lưu 1 ảnh.
        """
        self.output_dir = output_dir
        self.min_interval = min_interval
        self._last_save_time = 0.0

        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def draw_banner(frame, text: str = "INTRUDER DETECTED"):
        """Vẽ dải cảnh báo trên đầu frame."""
        h, w, _ = frame.shape

        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 255), thickness=-1)

        cv2.putText(
            frame,
            text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            lineType=cv2.LINE_AA,
        )

    def maybe_save_frame(self, frame):
        """Lưu frame nếu đã qua đủ min_interval giây."""
        now = time.time()
        if now - self._last_save_time < self.min_interval:
            return None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"alert_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        self._last_save_time = now
        return filename
