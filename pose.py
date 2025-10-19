from ultralytics import YOLO


model = YOLO("yolo11n.pt")  
model = YOLO("yolo11n-seg.pt")  
model = YOLO("yolo11n-pose.pt")  
model = YOLO("/home/farhod/Desktop/KAx 23/yolo11n-pose.pt")  


results = model.track("/home/farhod/Desktop/KAx 23/walk_1.mp4", show=True)  
results = model.track("/home/farhod/Desktop/KAx 23/walk_1.mp4", show=True, tracker="bytetrack.yaml")  

