import cv2
from ultralytics import YOLO

# Load YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")

# Open the laptop webcam
cap = cv2.VideoCapture(0)

# Set resolution to maximum available
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Create a full-screen window
cv2.namedWindow("AI Camera - Object Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("AI Camera - Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

# Run YOLO object detection
results = model(frame)

object_count = 0  

# Draw bounding boxes on detected objects
  for r in results:
        for box in r.boxes:
            object_count += 1  
            label = model.names[int(box.cls[0])] 
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 255, 0), 2)

# Display object count on the screen
  cv2.putText(frame, f"Objects Detected: {object_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, (0, 0, 255), 3)  

# Display the AI camera feed in full screen
  cv2.imshow("AI Camera - Object Detection", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
     break
cap.release()
cv2.destroyAllWindows()
