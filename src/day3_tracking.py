import cv2
from ultralytics import YOLO
import torch
import random

# ── Config ──────────────────────────────────────
MODEL_PATH = r"D:\shiva\Projects\TRINETRA\models\yolov8s.pt"
VIDEO_PATH = r"D:\shiva\Projects\TRINETRA\videos\test_traffic_2.mp4"

TARGET_CLASSES = [0, 2, 3, 5, 7]
CLASS_NAMES    = {0:"Person", 2:"Car", 3:"Motorcycle", 5:"Bus", 7:"Truck"}
# ────────────────────────────────────────────────

print("Loading TRINETRA Tracking Engine...")
model  = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device.upper()}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Cannot open video!")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video  : {width}x{height} @ {fps:.1f}FPS")
print("\nPress Q to quit | S to screenshot")
print("=" * 50)

# ── Color generator — each ID gets a unique color ──
def get_id_color(track_id):
    random.seed(int(track_id) * 3)
    r = random.randint(100, 255)
    g = random.randint(100, 255)
    b = random.randint(100, 255)
    return (b, g, r)  # OpenCV uses BGR

# ── Store trails for each vehicle ──────────────────
# trails[id] = list of center points (last 30 frames)
trails = {}
MAX_TRAIL = 30

# ── Stats ───────────────────────────────────────────
frame_count    = 0
screenshot_count = 0
all_ids_seen   = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    frame_count += 1

    # ── Run YOLO with ByteTrack ──────────────────────
    results = model.track(
        frame,
        device=device,
        classes=TARGET_CLASSES,
        conf=0.4,
        iou=0.5,
        tracker="bytetrack.yaml",  # ByteTrack built-in
        persist=True,               # keep IDs across frames
        verbose=False
    )

    # ── Process detections ───────────────────────────
    active_ids = []

    for result in results:
        if result.boxes.id is None:
            continue  # no tracking yet this frame

        for i, box in enumerate(result.boxes):
            track_id = int(result.boxes.id[i])
            cls_id   = int(box.cls[0])
            conf     = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls_name = CLASS_NAMES.get(cls_id, "Unknown")
            color    = get_id_color(track_id)

            # Center point of this box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Update trail
            if track_id not in trails:
                trails[track_id] = []
            trails[track_id].append((cx, cy))
            if len(trails[track_id]) > MAX_TRAIL:
                trails[track_id].pop(0)

            active_ids.append(track_id)
            all_ids_seen.add(track_id)

            # ── Draw trail (movement path) ────────────
            # pts = trails[track_id]
            # for j in range(1, len(pts)):
            #     alpha = j / len(pts)  # fade older points
            #     trail_color = tuple(int(c * alpha) for c in color)
            #     # cv2.line(frame, pts[j-1], pts[j], trail_color, 2)

            # ── Draw bounding box ─────────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ── Draw corner accents (pro look) ────────
            corner_len = 12
            thick = 3
            # Top-left
            cv2.line(frame,(x1,y1),(x1+corner_len,y1),color,thick)
            cv2.line(frame,(x1,y1),(x1,y1+corner_len),color,thick)
            # Top-right
            cv2.line(frame,(x2,y1),(x2-corner_len,y1),color,thick)
            cv2.line(frame,(x2,y1),(x2,y1+corner_len),color,thick)
            # Bottom-left
            cv2.line(frame,(x1,y2),(x1+corner_len,y2),color,thick)
            cv2.line(frame,(x1,y2),(x1,y2-corner_len),color,thick)
            # Bottom-right
            cv2.line(frame,(x2,y2),(x2-corner_len,y2),color,thick)
            cv2.line(frame,(x2,y2),(x2,y2-corner_len),color,thick)

            # ── Draw ID label ─────────────────────────
            label = f"ID:{track_id} {cls_name} {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame,
                (x1, y1 - lh - 10),
                (x1 + lw + 8, y1),
                color, -1)
            cv2.putText(frame, label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1)

            # ── Draw center dot ────────────────────────
            # cv2.circle(frame, (cx, cy), 4, color, -1)

    # ── Clean up trails of gone vehicles ─────────────
    gone_ids = set(trails.keys()) - set(active_ids)
    for gid in gone_ids:
        if len(trails[gid]) > 0:
            trails[gid].pop(0)
        if len(trails[gid]) == 0:
            del trails[gid]
            break

    # ── HUD Panel ────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 130), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "TRINETRA  TRACKING",
        (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 212, 200), 2)

    cv2.putText(frame,
        f"Frame      : {frame_count}/{total}",
        (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (200,200,200), 1)

    cv2.putText(frame,
        f"Active IDs : {len(active_ids)}",
        (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (0, 255, 180), 1)

    cv2.putText(frame,
        f"Total seen : {len(all_ids_seen)}",
        (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (200,200,200), 1)

    cv2.putText(frame,
        f"Device     : {device.upper()}",
        (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (200,200,200), 1)

    cv2.putText(frame, "Q=Quit  S=Screenshot",
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4, (80,80,80), 1)

    cv2.imshow("TRINETRA — Vehicle Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Stopped by user.")
        break
    elif key == ord('s'):
        screenshot_count += 1
        path = (f"D:\\shiva\\Projects\\TRINETRA\\outputs\\"
                f"track_{screenshot_count}.jpg")
        cv2.imwrite(path, frame)
        print(f"Saved: {path}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print(f"  Frames processed : {frame_count}")
print(f"  Unique vehicles  : {len(all_ids_seen)}")
print(f"  Screenshots saved: {screenshot_count}")
print("\n  Day 3 COMPLETE — Tracking is working!")
print("=" * 50)