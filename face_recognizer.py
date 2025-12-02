import os
import json
import cv2


class FaceRecognizer:
    """
    Nhận diện người quen / người lạ bằng LBPH + Haar cascade.
    - model_path: Models/face_lbph.xml (train từ train_face_recognizer.py)
    - labels_path: Models/face_labels.json
    - threshold: ngưỡng confidence, nhỏ hơn -> coi là người quen
    """

    def __init__(
        self,
        model_path: str = "Models/face_lbph.xml",
        labels_path: str = "Models/face_labels.json",
        threshold: float = 70.0,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Không tìm thấy labels: {labels_path}")

        # LBPH
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(model_path)

        # map id -> tên
        with open(labels_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            self.id_to_name = {int(k): v for k, v in raw.items()}

        self.threshold = threshold

        # Haar cascade để tìm khuôn mặt
        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        self.face_cascade = cv2.CascadeClassifier(frontal_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Không load được cascade: {frontal_path}")

    def recognize(self, frame):
        """
        Nhận diện tất cả khuôn mặt trong frame.
        Trả về list dict:
        {
            "box": (x, y, w, h),
            "name": "Phuc" hoặc "Unknown",
            "confidence": float,
            "is_known": bool
        }
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(60, 60),
        )

        results = []
        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi_resized = cv2.resize(roi, (200, 200))

            label_id, confidence = self.recognizer.predict(roi_resized)

            if confidence < self.threshold and label_id in self.id_to_name:
                name = self.id_to_name[label_id]
                is_known = True
            else:
                name = "Unknown"
                is_known = False

            results.append(
                {
                    "box": (x, y, w, h),
                    "name": name,
                    "confidence": float(confidence),
                    "is_known": is_known,
                }
            )

        return results

    def draw_faces(self, frame, results):
        """
        Vẽ khung + tên + confidence lên frame dựa trên kết quả recognize().
        """
        for r in results:
            x, y, w, h = r["box"]
            name = r["name"]
            conf = r["confidence"]
            is_known = r["is_known"]

            color = (0, 255, 0) if is_known else (0, 0, 255)
            label = f"{name} ({conf:.1f})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
