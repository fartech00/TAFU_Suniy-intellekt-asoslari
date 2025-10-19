
from ultralytics import YOLO
import cv2


video_path = "/home/farhod/Desktop/KAx 23/walk_1.mp4"    
output_path = "/home/farhod/Desktop/KAx 23/natija/natija.mp4"
model_path = "/home/farhod/Desktop/KAx 23/yolo11m-seg.pt"         

target_classes = ["human", "car"]  


model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        class_name = model.names[int(cls_id)]
        if class_name in target_classes:
            x1, y1, x2, y2 = map(int, box)
            # Draw bounding box
            color = (0, 255, 0) if class_name == "inson" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)

    cv2.imshow("Kuzatish", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Natija saqlandi:", output_path)
