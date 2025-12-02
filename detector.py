from ultralytics import YOLO
import cv2


class personDetector:
    """
    Dùng YOLOv8n chỉ để detect người (class 0).
    img_size để 320 cho nhẹ, đủ cho bài này.
    """

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5, img_size=320):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.img_size = img_size

    def detect(self, frame):
        """
        Trả về list bbox người: [(x1, y1, x2, y2, conf), ...]
        """
        results = self.model(
            frame,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            classes=[0],       # chỉ class person
            verbose=False,
            device="cpu",
        )

        dets = []
        r0 = results[0]
        if r0.boxes is None:
            return dets

        for box in r0.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            dets.append((int(x1), int(y1), int(x2), int(y2), conf))

        return dets

    def draw_detections(self, frame, detections):
        """
        Vẽ khung người cho debug.
        """
        for (x1, y1, x2, y2, conf) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Person {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
