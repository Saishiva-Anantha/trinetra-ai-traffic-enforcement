import cv2
from ultralytics import YOLO
import torch

# ── Config ──────────────────────────────────────
MODEL_PATH  = r"D:\shiva\Projects\TRINETRA\models\yolov8s.pt"
VIDEO_PATH  = r"D:\shiva\Projects\TRINETRA\videos\test_traffic.mp4"

# Classes we care about (from COCO dataset)
# 2=car, 3=motorcycle, 5=bus, 7=truck, 0=person
TARGET_CLASSES = [0, 2, 3, 5, 7]
CLASS_NAMES    = {0:"Person", 2:"Car", 3:"Motorcycle", 5:"Bus", 7:"Truck"}

# Box colors per class (BGR format for OpenCV)
COLORS = {
    0: (0, 255, 255),    # Person  → Yellow
    2: (0, 212, 200),    # Car     → Cyan
    3: (0, 100, 255),    # Moto    → Orange
    5: (255, 100, 0),    # Bus     → Blue
    7: (100, 255, 100),  # Truck   → Green
}
# ────────────────────────────────────────────────

print("Loading TRINETRA detection engine...")
model  = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video file!")
    print(f"Check path: {VIDEO_PATH}")
    exit()

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps:.1f}FPS | {total} frames")
print("\nPress Q to quit | Press S to save screenshot")
print("=" * 45)

frame_count = 0
screenshot_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    frame_count += 1

    # Run YOLO detection
    results = model(
        frame,
        device=device,
        classes=TARGET_CLASSES,  # only detect our classes
        conf=0.4,                 # confidence threshold
        verbose=False
    )

    # Count detections per class
    counts = {name: 0 for name in CLASS_NAMES.values()}

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls_name = CLASS_NAMES.get(cls_id, "Unknown")
            color    = COLORS.get(cls_id, (255, 255, 255))

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{cls_name} {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame,
                (x1, y1 - lh - 8), (x1 + lw + 6, y1),
                color, -1)

            # Draw label text
            cv2.putText(frame, label,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1)

            # Update counts
            if cls_name in counts:
                counts[cls_name] += 1

    # ── HUD overlay (top-left info panel) ───────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, "TRINETRA v1.0",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 212, 200), 2)

    cv2.putText(frame, f"Frame: {frame_count}/{total}",
        (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (180, 180, 180), 1)

    y = 70
    for name, count in counts.items():
        if count > 0:
            cv2.putText(frame, f"{name}: {count}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)
            y += 20

    cv2.putText(frame, "Q=Quit  S=Screenshot",
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4, (100, 100, 100), 1)
    # ─────────────────────────────────────────────

    # Show frame
    cv2.imshow("TRINETRA — Live Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Stopped by user.")
        break
    elif key == ord('s'):
        screenshot_count += 1
        path = (f"D:\\shiva\\Projects\\TRINETRA\\outputs\\"
                f"screenshot_{screenshot_count}.jpg")
        cv2.imwrite(path, frame)
        print(f"Screenshot saved: {path}")

cap.release()
cv2.destroyAllWindows()

print(f"\nSession summary:")
print(f"  Frames processed : {frame_count}")
print(f"  Screenshots saved: {screenshot_count}")
print("\nDay 2 COMPLETE!")