from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO(r"C:\Users\trong\Desktop\NoctisAI - Detection\server\pothole-detection-camera\best.pt")

class_names = model.names

video_path = r"C:\Users\trong\Desktop\NoctisAI - Detection\server\pothole-detection-camera\pothole.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở video")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        print("Đã đến cuối video hoặc không thể đọc khung hình.")
        break

    results = model.predict(img)

    for r in results:
        boxes = r.boxes
        masks = r.masks

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (img.shape[1], img.shape[0]))
                contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Video Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
