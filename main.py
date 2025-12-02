import cv2

from camera_stream import camera_Stream
from detector import personDetector
from alert_manager import AlertManager


def main():
    # Khởi tạo các thành phần
    detector = personDetector(
        model_path="yolov8n.pt",   # sau này đổi thành models/best.pt nếu fine-tune
        conf_threshold=0.5,
        img_size=640,
    )
    alerts = AlertManager(output_dir="alerts", min_interval=3.0)
    camera = camera_Stream(device_index=0)

    print("=== Hệ thống phát hiện người & cảnh báo ===")
    print("Nhấn 'q' để thoát.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Không đọc được frame từ camera, dừng.")
                break

            # Phát hiện người
            detections = detector.detect(frame)

            # Vẽ bounding box
            detector.draw_detections(frame, detections)

            # Nếu có người → bật cảnh báo
            if len(detections) > 0:
                alerts.draw_banner(frame, text="People")
                saved_path = alerts.maybe_save_frame(frame)
                if saved_path is not None:
                    print(f"[ALERT] Đã lưu ảnh cảnh báo: {saved_path}")

            # Hiển thị
            cv2.imshow("Intruder Alert Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
