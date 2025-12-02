import cv2
import threading
import numpy as np  # thêm để xử lý ảnh

from camera_stream import camera_Stream
from detector import personDetector
from alert_manager import AlertManager
from face_recognizer import FaceRecognizer

try:
    from notifier import TelegramNotifier
except ImportError:
    TelegramNotifier = None


def create_notifier():
    if TelegramNotifier is None:
        print("[INFO] Chưa có notifier.py hoặc không import được, bỏ qua Telegram.")
        return None

    TELEGRAM_BOT_TOKEN = "7770244537:AAHCaPGU-_5E7D1EZancGITXqt_OZ6idtBI"
    TELEGRAM_CHAT_ID = "6269445809"

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[INFO] TELEGRAM_BOT_TOKEN / CHAT_ID chưa được cấu hình, bỏ qua Telegram.")
        return None

    return TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)


def main():
    # --- Khởi tạo các module ---
    person_detector = personDetector(
        model_path="yolov8n.pt",   # hoặc "models/best.pt" nếu ông fine-tune
        conf_threshold=0.5,
        img_size=320,
    )

    # Nhận diện người quen / người lạ bằng LBPH
    face_recognizer = FaceRecognizer(
        model_path="Models/face_lbph.xml",
        labels_path="Models/face_labels.json",
        threshold=80.0,
    )

    alerts = AlertManager(output_dir="alerts", min_interval=8.0)

    # Camera laptop hoặc Pi
    camera = camera_Stream(device_index=0)

    # Notifier (Telegram)
    notifier = create_notifier()


    unknown_streak = 0
    UNKNOWN_STREAK_LIMIT = 10

    # Đếm frame để skip YOLO
    frame_idx = 0
    last_person_dets = []

    print("=== Hệ thống nhận diện người quen / người lạ + cảnh báo ===")
    print("Nhấn 'q' để thoát.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Không đọc được frame từ camera, dừng.")
                break

            frame_idx += 1

            # 1. Phát hiện người (body) -> CHỈ CHẠY YOLO MỖI 3 FRAME
            if frame_idx % 3 == 0:
                person_dets = person_detector.detect(frame)
                last_person_dets = person_dets
            else:
                person_dets = last_person_dets

            person_detector.draw_detections(frame, person_dets)

            # 2. Nhận diện mặt (known/unknown)
            face_results = face_recognizer.recognize(frame)
            face_recognizer.draw_faces(frame, face_results)

            # 3. Quyết định trạng thái
            any_known = any(f["is_known"] for f in face_results)
            any_unknown = any(not f["is_known"] for f in face_results)
            any_person = len(person_dets) > 0

            # cập nhật chuỗi frame unknown
            if any_unknown or (any_person and not any_known):
                unknown_streak += 1
            else:
                unknown_streak = 0

            status_text = "NO PERSON"

            # 3.1. Nếu unknown nhiều frame liên tiếp -> chắc chắn xâm nhập
            if unknown_streak >= UNKNOWN_STREAK_LIMIT:
                status_text = "INTRUDER (Unknown / body)"
                alerts.draw_banner(frame, text="INTRUDER DETECTED")

                saved_path = alerts.maybe_save_frame(frame)
                if saved_path is not None:
                    print(f"[ALERT] Đã lưu ảnh cảnh báo: {saved_path}")
                    if notifier is not None:
                        # GỬI TELEGRAM Ở THREAD RIÊNG ĐỂ KHỎI GIẬT
                        threading.Thread(
                            target=notifier.send_alert,
                            args=("Cảnh báo: phát hiện người lạ trước camera!", saved_path),
                            daemon=True,
                        ).start()

            # 3.2. Có ít nhất một người quen, không có unknown
            elif any_known:
                known_names = {f["name"] for f in face_results if f["is_known"]}
                status_text = "KNOWN: " + ", ".join(known_names)

                # Vẽ dải xanh báo an toàn
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, 0), (w, 40), (0, 150, 0), thickness=-1)
                cv2.putText(
                    frame,
                    status_text,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            # 3.3. Thấy body nhưng không thấy mặt rõ
            elif any_person:
                status_text = "INTRUDER (Body only)"
                alerts.draw_banner(frame, text="INTRUDER DETECTED")

                saved_path = alerts.maybe_save_frame(frame)
                if saved_path is not None:
                    print(f"[ALERT] Body detected, đã lưu ảnh: {saved_path}")
                    if notifier is not None:
                        threading.Thread(
                            target=notifier.send_alert,
                            args=("Cảnh báo: phát hiện người (body) nhưng không nhận diện được mặt!", saved_path),
                            daemon=True,
                        ).start()

            # 4. Hiển thị trạng thái ở góc màn hình (trên frame gốc)
            cv2.putText(
                frame,
                status_text,
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )


            # ===== THU NHỎ BỀ RỘNG CỬA SỔ HIỂN THỊ =====
            display = frame.copy()
            h, w = display.shape[:2]

            max_width = 800  # muốn nhỏ nữa thì giảm xuống 700, 640,...
            if w > max_width:
                scale = max_width / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                display = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_AREA)

            cv2.imshow("Face-based Intruder System", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Đã tắt camera & cửa sổ hiển thị.")
if __name__ == "__main__":
    main()