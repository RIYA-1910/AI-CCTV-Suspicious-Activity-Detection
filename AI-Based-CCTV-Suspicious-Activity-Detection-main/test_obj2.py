"""
Diagnostic: test object detection on your video.
Run from cctv_project folder:
    python test_obj2.py "data\test.mp4"
"""
import sys
import cv2
from ultralytics import YOLO

if len(sys.argv) < 2:
    print("Usage: python test_obj2.py <video_path>")
    sys.exit(1)

VIDEO = sys.argv[1]

# --- Step 1: check OpenCV can open the video ---
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"ERROR: OpenCV cannot open '{VIDEO}'")
    print("Check the path is correct and the file is not corrupted.")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video opened OK: {w}x{h}, {fps:.1f} fps, {total_frames} frames")

# Read the 10th frame (skip the very first which is sometimes blank)
for _ in range(10):
    ok, frame = cap.read()
cap.release()

if not ok or frame is None:
    print("ERROR: Could not read frame from video (file may be corrupt or empty).")
    sys.exit(1)

print(f"Frame read OK: shape={frame.shape}")

# Save the frame so you can see what YOLO is looking at
cv2.imwrite("test_frame.jpg", frame)
print("Saved test_frame.jpg - open it to confirm the video loaded correctly.")

# --- Step 2: run object detection ---
CLASSES = {24:"backpack", 25:"umbrella", 26:"handbag", 28:"suitcase",
           39:"bottle",   41:"cup",      63:"laptop",  67:"cell phone", 73:"book"}

print("\nLoading yolov8s.pt ...")
model = YOLO("yolov8s.pt")
print("Running detection (conf=0.15, very low to catch anything) ...")
res = model.predict(frame, classes=list(CLASSES.keys()), conf=0.15, verbose=False)[0]

print(f"\nDetections found: {len(res.boxes)}")
if len(res.boxes) == 0:
    print("No objects detected in this frame.")
    print("Try a frame where a bag/bottle is clearly visible.")
else:
    for box, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
        name = CLASSES[int(cls)]
        print(f"  {name:12s}  conf={float(conf):.2f}  box={[round(float(v)) for v in box]}")

# Also run with NO class filter to see ALL detections
print("\nAll YOLO detections (no class filter, conf=0.3):")
res2 = model.predict(frame, conf=0.3, verbose=False)[0]
names = model.names
print(f"  Total: {len(res2.boxes)}")
for box, cls, conf in zip(res2.boxes.xyxy, res2.boxes.cls, res2.boxes.conf):
    print(f"  {names[int(cls)]:15s}  conf={float(conf):.2f}")
