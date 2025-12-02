import cv2


class camera_Stream:
    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(device_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được camera index {device_index}")

        # THỬ FULL HD NHẸ: 1280x720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
