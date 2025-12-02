import os
import json
import cv2

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "face_lbph.xml")
LABELS_PATH = os.path.join(MODELS_DIR, "face_labels.json")


def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        raise RuntimeError("Chưa có model/labels, hãy chạy train_face_recognizer.py trước.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        id_to_name = {int(k): v for k, v in json.load(f).items()}

    frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"

    frontal_cascade = cv2.CascadeClassifier(frontal_path)
    profile_cascade = cv2.CascadeClassifier(profile_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được camera")
        return

    threshold = 80.0  # tăng/giảm sau nếu cần

    print("Test nhận diện mặt. Nhấn 'q' để thoát.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = frontal_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (200, 200))

            label_id, confidence = recognizer.predict(face_resized)

            if confidence < threshold and label_id in id_to_name:
                name = id_to_name[label_id]
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            text = f"{name} ({confidence:.1f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        cv2.imshow("Face Recognition Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
