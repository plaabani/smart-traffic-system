import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")   # small & fast model

# Load video
cap = cv2.VideoCapture("video.mp4")  # or 0 for webcam

# Vehicle classes (COCO dataset IDs)
vehicle_classes = [2, 3, 5, 7]  
# 2=car, 3=motorbike, 5=bus, 7=truck

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    vehicle_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls in vehicle_classes:
                vehicle_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = "Vehicle"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # 🚦 Traffic Signal Logic
    if vehicle_count < 5:
        signal = "GREEN (Low Traffic)"
        time = 10
    elif vehicle_count < 15:
        signal = "YELLOW (Medium Traffic)"
        time = 20
    else:
        signal = "RED (High Traffic)"
        time = 30

    # Display info
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.putText(frame, f"Signal: {signal}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.putText(frame, f"Timer: {time}s", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Traffic System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()