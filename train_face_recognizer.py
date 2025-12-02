import os
import json
import cv2
import numpy as np

FACES_DIR = "faces"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "face_lbph.xml")
LABELS_PATH = os.path.join(MODELS_DIR, "face_labels.json")


def load_face_dataset():
    images = []
    labels = []
    id_to_name = {}
    current_id = 0

    if not os.path.isdir(FACES_DIR):
        raise RuntimeError(f"Thư mục '{FACES_DIR}' không tồn tại.")

    for name in sorted(os.listdir(FACES_DIR)):
        person_dir = os.path.join(FACES_DIR, name)
        if not os.path.isdir(person_dir):
            continue

        print(f"Đọc dữ liệu của: {name}")
        id_to_name[current_id] = name

        for filename in os.listdir(person_dir):
            path = os.path.join(person_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  Bỏ qua file (không đọc được): {path}")
                continue

            # chuẩn về 200x200
            img_resized = cv2.resize(img, (200, 200))
            images.append(img_resized)
            labels.append(current_id)

        current_id += 1

    if not images:
        raise RuntimeError("Không có ảnh nào trong 'faces/'.")

    return images, labels, id_to_name


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    images, labels, id_to_name = load_face_dataset()

    print(f"Tổng ảnh: {len(images)}, số người: {len(id_to_name)}")

    images_np = np.array(images, dtype=np.uint8)
    labels_np = np.array(labels, dtype=np.int32)

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )

    print("Train LBPH...")
    recognizer.train(images_np, labels_np)
    recognizer.save(MODEL_PATH)
    print(f"Đã lưu model: {MODEL_PATH}")

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(id_to_name, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu label map: {LABELS_PATH}")


if __name__ == "__main__":
    main()
