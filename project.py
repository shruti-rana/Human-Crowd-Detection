import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Open video
cap = cv2.VideoCapture("croud.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect only people (class 0)
    results = model(frame, classes=[0])

    people_count = 0

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Label
            cv2.putText(frame, "Person",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0),
                        2)

            people_count += 1

    # Display total people
    cv2.putText(frame,
                f"People Count: {people_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2)

    cv2.imshow("People Detection", frame)

    if cv2.waitKey(1) == 27:   # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
