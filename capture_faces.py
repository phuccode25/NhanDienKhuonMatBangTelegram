import os
import cv2


def detect_best_face(gray, frontal_cascade, profile_cascade):
    h, w = gray.shape[:2]
    cx, cy = w // 2, h // 2
    candidates = []
    # Mặt thẳng / hơi nghiêng
    frontal_faces = frontal_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,   # nhạy hơn 1.1
        minNeighbors=4,
        minSize=(60, 60),
    )
    for (x, y, fw, fh) in frontal_faces:
        candidates.append((x, y, fw, fh))
    #Mặt nghiêng phải (profile trong ảnh gốc)
    profile_right = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(60, 60),
    )
    for (x, y, fw, fh) in profile_right:
        candidates.append((x, y, fw, fh))
    #  Mặt nghiêng trái (detect trên ảnh lật)
    gray_flip = cv2.flip(gray, 1)
    profile_left = profile_cascade.detectMultiScale(
        gray_flip,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(60, 60),
    )
    for (x, y, fw, fh) in profile_left:
        # chuyển về tọa độ ảnh gốc
        x_orig = w - x - fw
        candidates.append((x_orig, y, fw, fh))

    if not candidates:
        return None
    # Chọn mặt có tâm gần giữa khung hình nhất
    def center_dist2(face):
        x, y, fw, fh = face
        fx = x + fw // 2
        fy = y + fh // 2
        return (fx - cx) ** 2 + (fy - cy) ** 2

    best_face = min(candidates, key=center_dist2)
    return best_face


def main():
    person_name = input("Nhập tên (không dấu, không space): ").strip()
    if not person_name:
        print("Tên trống, thoát.")
        return

    save_dir = os.path.join("faces", person_name)
    os.makedirs(save_dir, exist_ok=True)

    # Cascade mặt thẳng + mặt nghiêng
    frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
    print("Frontal cascade:", frontal_path)
    print("Profile cascade:", profile_path)

    frontal_cascade = cv2.CascadeClassifier(frontal_path)
    profile_cascade = cv2.CascadeClassifier(profile_path)

    if frontal_cascade.empty() or profile_cascade.empty():
        raise RuntimeError("Không load được cascade mặt, kiểm tra file trong cv2/data.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được camera")
        return

    target_count = 40
    saved_count = 0

    print("Đưa mặt vào khung vàng, quay trái/phải, cúi/ngẩng.")
    print("Bấm SPACE hoặc 'c' để chụp 1 ảnh, 'q' để thoát.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Không đọc được frame, dừng.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        cx, cy = w // 2, h // 2

        # tìm mặt tốt nhất (thẳng hoặc nghiêng)
        best_face = detect_best_face(gray, frontal_cascade, profile_cascade)

        if best_face is None:
            cv2.putText(
                frame,
                "Khong thay mat ro, lai gan hon / quay nhe lai.",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        else:
            x, y, fw, fh = best_face
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 200, 0), 2)

        # khung hướng dẫn ở giữa
        guide_w, guide_h = w // 3, h // 3
        gx1, gy1 = cx - guide_w // 2, cy - guide_h // 2
        gx2, gy2 = cx + guide_w // 2, cy + guide_h // 2
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 255), 1)

        # text progress
        cv2.putText(
            frame,
            f"{person_name}: {saved_count}/{target_count}",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "[SPACE/c] chup | q thoat",
            (10, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Capture Faces", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Bấm SPACE hoặc 'c' để chụp
        if key in (ord(" "), ord("c")) and best_face is not None:
            if saved_count < target_count:
                x, y, fw, fh = best_face
                face_roi = gray[y:y + fh, x:x + fw]
                face_resized = cv2.resize(face_roi, (200, 200))

                filename = os.path.join(
                    save_dir,
                    f"{person_name}_{saved_count:03d}.jpg"
                )
                cv2.imwrite(filename, face_resized)
                saved_count += 1
                print(f"Đã lưu {filename}")

        if saved_count >= target_count:
            print("Đã đủ ảnh, thoát.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
