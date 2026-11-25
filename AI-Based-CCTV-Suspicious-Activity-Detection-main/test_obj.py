from ultralytics import YOLO
import cv2, sys

VIDEO = sys.argv[1]   # pass your video path as argument
cap = cv2.VideoCapture(VIDEO)
ok, frame = cap.read()
cap.release()

CLASSES = {24:"backpack",25:"umbrella",26:"handbag",28:"suitcase",
           39:"bottle",41:"cup",63:"laptop",67:"cell phone",73:"book"}

model = YOLO("yolov8s.pt")
res = model.predict(frame, classes=list(CLASSES.keys()), conf=0.2, verbose=True)[0]

print(f"Detections: {len(res.boxes)}")
for box, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
    print(f"  {CLASSES[int(cls)]}  conf={float(conf):.2f}  box={[round(float(v)) for v in box]}")