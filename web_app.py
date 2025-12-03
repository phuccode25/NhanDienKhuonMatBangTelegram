# web_app.py
import cv2
import threading
from flask import Flask, Response, render_template

from camera_stream import camera_Stream
from detector import personDetector
from alert_manager import AlertManager
from face_recognizer import FaceRecognizer

try:
    from notifier import TelegramNotifier
except ImportError:
    TelegramNotifier = None


app = Flask(__name__)


def create_notifier():
    if TelegramNotifier is None:
        print("[INFO] Chưa có notifier.py hoặc không import được, bỏ qua Telegram.")
        return None

    # !!! NHỚ: trước khi up code public thì giấu token đi !!!
    TELEGRAM_BOT_TOKEN = "7770244537:AAHCaPGU-_5E7D1EZancGITXqt_OZ6idtBI"
    TELEGRAM_CHAT_ID = "6269445809"

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[INFO] TELEGRAM_BOT_TOKEN / CHAT_ID chưa cấu hình, bỏ qua Telegram.")
        return None

    return TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)


# ======== KHỞI TẠO CÁC MODULE GIỐNG main() ========
person_detector = personDetector(
    model_path="yolov8n.pt",
    conf_threshold=0.5,
    img_size=320,
)

face_recognizer = FaceRecognizer(
    model_path="Models/face_lbph.xml",
    labels_path="Models/face_labels.json",
    threshold=80.0,
)

alerts = AlertManager(output_dir="alerts", min_interval=8.0)
camera = camera_Stream(device_index=0)
notifier = create_notifier()

# state dùng chung giữa các frame
UNKNOWN_STREAK_LIMIT = 10
unknown_streak = 0
frame_idx = 0
last_person_dets = []


def process_frame(frame):
    """
    Toàn bộ logic nhận diện + cảnh báo của bạn,
    chỉ bỏ phần imshow/waitKey đi.
    """
    global unknown_streak, frame_idx, last_person_dets

    frame_idx += 1

    # 1. Phát hiện người (body) – YOLO, skip mỗi 3 frame
    if frame_idx % 3 == 0:
        person_dets = person_detector.detect(frame)
        last_person_dets = person_dets
    else:
        person_dets = last_person_dets

    person_detector.draw_detections(frame, person_dets)

    # 2. Nhận diện mặt (known / unknown)
    face_results = face_recognizer.recognize(frame)
    face_recognizer.draw_faces(frame, face_results)

    # 3. Quyết định trạng thái
    any_known = any(f["is_known"] for f in face_results)
    any_unknown = any(not f["is_known"] for f in face_results)
    any_person = len(person_dets) > 0

    if any_unknown or (any_person and not any_known):
        unknown_streak += 1
    else:
        unknown_streak = 0

    status_text = "NO PERSON"

    # 3.1. Unknown nhiều frame liên tiếp → xâm nhập
    if unknown_streak >= UNKNOWN_STREAK_LIMIT:
        status_text = "INTRUDER (Unknown / body)"
        alerts.draw_banner(frame, text="INTRUDER DETECTED")

        saved_path = alerts.maybe_save_frame(frame)
        if saved_path is not None:
            print(f"[ALERT] Đã lưu ảnh cảnh báo: {saved_path}")
            if notifier is not None:
                threading.Thread(
                    target=notifier.send_alert,
                    args=("Cảnh báo: phát hiện người lạ trước camera!", saved_path),
                    daemon=True,
                ).start()

    # 3.2. Có ít nhất 1 người quen, không có unknown
    elif any_known:
        known_names = {f["name"] for f in face_results if f["is_known"]}
        status_text = "KNOWN: " + ", ".join(known_names)

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

    # 3.3. Thấy body nhưng không nhận diện được mặt
    elif any_person:
        status_text = "INTRUDER (Body only)"
        alerts.draw_banner(frame, text="INTRUDER DETECTED")

        saved_path = alerts.maybe_save_frame(frame)
        if saved_path is not None:
            print(f"[ALERT] Body detected, đã lưu ảnh: {saved_path}")
            if notifier is not None:
                threading.Thread(
                    target=notifier.send_alert,
                    args=(
                        "Cảnh báo: phát hiện người (body) nhưng không nhận diện được mặt!",
                        saved_path,
                    ),
                    daemon=True,
                ).start()

    # 4. Vẽ trạng thái ở dưới
    cv2.putText(
        frame,
        status_text,
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    # 5. Thu nhỏ cho nhẹ khi stream
    h, w = frame.shape[:2]
    max_width = 800
    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return frame


def gen_frames():
    """Đọc camera_Stream + xử lý + trả về MJPEG cho <img>."""
    while True:
        ok, frame = camera.read()
        if not ok:
            continue

        frame = process_frame(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")  # giao diện đẹp lúc nãy


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
