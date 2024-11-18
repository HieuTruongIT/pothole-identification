from ultralytics import YOLO
import cv2
import numpy as np


model = YOLO(r"C:\Users\trong\Desktop\NoctisAI - Detection\server\pothole-detection-camera\best.pt")
class_names = model.names

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

output_file_path = r"C:\Users\trong\Desktop\NoctisAI - Detection\server\pothole-detection-camera\real-time-using-yolov8.txt"

with open(output_file_path, "w") as file:
    file.write("Pothole Detection Log\n")
    file.write("Format: [Class, Width, Height, Damage Ratio]\n\n")

count = 0

while True:
    ret, img = cap.read()
    if not ret:
        print("Không thể nhận khung hình từ camera, thoát...")
        break

    count += 1
    if count % 3 != 0:  
        continue

    img = cv2.resize(img, (1020, 500))  
    h, w, _ = img.shape


    results = model.predict(img)

    for r in results:
        boxes = r.boxes  
        masks = r.masks  

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    d = int(box.cls)  
                    c = class_names[d] 
                    x, y, width, height = cv2.boundingRect(contour)  

                    mask_area = np.sum(seg)
                    bbox_area = width * height
                    damage_ratio = mask_area / bbox_area if bbox_area > 0 else 0

                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    cv2.putText(img, f"{c}: {damage_ratio:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    with open(output_file_path, "a") as file:
                        file.write(f"[{c}, Width: {width}, Height: {height}, Damage Ratio: {damage_ratio:.2f}]\n")

    cv2.imshow('Camera Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
