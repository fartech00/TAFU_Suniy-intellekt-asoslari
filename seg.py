
from ultralytics import YOLO
import cv2

model = YOLO("/home/farhod/Desktop/KAx 23/yolo11m-seg.pt")


video_path = "/home/farhod/Desktop/KAx 23/walk_1.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "/home/farhod/Desktop/KAx 23/output_seg_video.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


while True:
    ret, frame = cap.read()
    if not ret:
        break
   
    results = model(frame)[0]
    annotated_frame = results.plot()
    
    cv2.imshow("YOLO Segmentation", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Saqlandi:", output_path)
