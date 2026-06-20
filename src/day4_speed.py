import cv2
from ultralytics import YOLO
import torch
import random
import numpy as np
from collections import defaultdict, deque
import os
from datetime import datetime
import csv

# ── Config ──────────────────────────────────────────
MODEL_PATH = r"D:\shiva\Projects\TRINETRA\models\yolov8s.pt"
VIDEO_PATH = r"D:\shiva\Projects\TRINETRA\videos\test_traffic_2.mp4"

TARGET_CLASSES = [0, 2, 3, 5, 7]
CLASS_NAMES    = {0:"Person", 2:"Car", 3:"Motorcycle",
                  5:"Bus",    7:"Truck"}

# ── Speed thresholds (KM/H) ─────────────────────────
SPEED_LIMIT_HIGHWAY = 80   # above this = violation
SPEED_LIMIT_URBAN   = 40

# ── Calibration ─────────────────────────────────────
# These 2 lines are drawn on the video frame.
# Measure the pixel gap between them, then set the
# real-world distance they represent in meters.
# We will auto-calculate from video height.
# You can tune REAL_DISTANCE_METERS for your video.
REAL_DISTANCE_METERS = 10.0   # tweak this for accuracy

# ── Smoothing ────────────────────────────────────────
SPEED_SMOOTH_FRAMES = 8   # average over last N frames
# ─────────────────────────────────────────────────────

print("=" * 50)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(f"  TRINETRA — Speed Detection Engine ({timestamp})")
print("=" * 50)

model  = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device  : {device.upper()}")

# Create output folder for this run
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

OUTPUT_FOLDER = os.path.join(
    r"D:\shiva\Projects\TRINETRA\outputs",
    timestamp
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Output Folder : {OUTPUT_FOLDER}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Cannot open video!")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"  Video   : {width}x{height} @ {fps:.1f} FPS")

# ── Calibration lines (horizontal) ──────────────────
# Line A = 30% from top,  Line B = 70% from top
LINE_A_Y = int(height * 0.30)
LINE_B_Y = int(height * 0.70)
PIXEL_DISTANCE = LINE_B_Y - LINE_A_Y
PIXEL_TO_METER = REAL_DISTANCE_METERS / PIXEL_DISTANCE

print(f"  Calib   : {PIXEL_DISTANCE}px = "
      f"{REAL_DISTANCE_METERS}m → "
      f"{PIXEL_TO_METER:.4f} m/px")
print("\n  Press Q to quit | S to save screenshot")
print("  Press + / - to adjust speed limit")
print("=" * 50)

# ── Tracking data ────────────────────────────────────
# positions[id]  = deque of (cx, cy) last N frames
# speeds[id]     = deque of speed values for smoothing
# violations[id] = True if this ID exceeded limit
positions  = defaultdict(lambda: deque(maxlen=SPEED_SMOOTH_FRAMES))
speeds     = defaultdict(lambda: deque(maxlen=SPEED_SMOOTH_FRAMES))
violations = {}
speed_limit = SPEED_LIMIT_HIGHWAY

frame_count       = 0
screenshot_count  = 0
total_violations  = 0

def get_color(track_id):
    random.seed(int(track_id) * 3)
    return (random.randint(100,255),
            random.randint(100,255),
            random.randint(100,255))

def calculate_speed(track_id, cx, cy):
    """Calculate smoothed speed for a vehicle."""
    positions[track_id].append((cx, cy))

    if len(positions[track_id]) < 2:
        return 0.0

    # Distance moved over last N frames
    pts = list(positions[track_id])
    total_dist_px = 0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        total_dist_px += np.sqrt(dx**2 + dy**2)

    avg_dist_px      = total_dist_px / (len(pts) - 1)
    meters_per_frame = avg_dist_px * PIXEL_TO_METER
    meters_per_sec   = meters_per_frame * fps
    kmh              = meters_per_sec * 3.6

    # Smooth speed
    speeds[track_id].append(kmh)
    smoothed = sum(speeds[track_id]) / len(speeds[track_id])
    return round(smoothed, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("\nVideo ended.")
        break

    frame_count += 1

    # ── Draw calibration lines ───────────────────────
    # Line A (top)
    cv2.line(frame, (0, LINE_A_Y), (width, LINE_A_Y),
             (0, 255, 255), 1)
    cv2.putText(frame, "CALIB LINE A",
        (10, LINE_A_Y - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

    # Line B (bottom)
    cv2.line(frame, (0, LINE_B_Y), (width, LINE_B_Y),
             (0, 255, 255), 1)
    cv2.putText(frame, "CALIB LINE B",
        (10, LINE_B_Y + 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

    # ── Run YOLO + ByteTrack ─────────────────────────
    results = model.track(
        frame,
        device=device,
        classes=TARGET_CLASSES,
        conf=0.4,
        iou=0.5,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )

    active_count = 0

    for result in results:
        if result.boxes.id is None:
            continue

        for i, box in enumerate(result.boxes):
            track_id = int(result.boxes.id[i])
            cls_id   = int(box.cls[0])
            conf     = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])

            cls_name = CLASS_NAMES.get(cls_id, "?")
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Calculate speed
            kmh = calculate_speed(track_id, cx, cy)

            # Check violation
            is_violation = kmh > speed_limit and kmh > 5
            
            # ── Box color ────────────────────────────
            # Red = violation, Green = safe, Gray = slow
            if kmh < 5:
                color = (120, 120, 120)   # still/slow → gray
            elif is_violation:
                color = (0, 0, 255)       # violation  → red
            else:
                color = (0, 212, 0)       # safe       → green

            # ── Draw bounding box ────────────────────
            cv2.rectangle(frame,(x1,y1),(x2,y2), color, 2)

            # ── Corner accents ───────────────────────
            cl = 10
            tk = 2
            cv2.line(frame,(x1,y1),(x1+cl,y1),color,tk)
            cv2.line(frame,(x1,y1),(x1,y1+cl),color,tk)
            cv2.line(frame,(x2,y1),(x2-cl,y1),color,tk)
            cv2.line(frame,(x2,y1),(x2,y1+cl),color,tk)
            cv2.line(frame,(x1,y2),(x1+cl,y2),color,tk)
            cv2.line(frame,(x1,y2),(x1,y2-cl),color,tk)
            cv2.line(frame,(x2,y2),(x2-cl,y2),color,tk)
            cv2.line(frame,(x2,y2),(x2,y2-cl),color,tk)

            # ── Speed label ──────────────────────────
            if kmh > 5:
                warn  = " !" if is_violation else ""
                label = f"ID:{track_id} {cls_name}{warn}"
                spd_l = f"{kmh} KM/H"
            else:
                label = f"ID:{track_id} {cls_name}"
                spd_l = "---"

            # Label background
            (lw,lh),_ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame,
                (x1, y1-lh-20),(x1+max(lw,70)+6, y1),
                color, -1)

            # Class + ID
            cv2.putText(frame, label,
                (x1+3, y1-12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0,0,0), 1)

            # Speed value
            spd_color = (0,0,180) if is_violation else (0,80,0)
            cv2.putText(frame, spd_l,
                (x1+3, y1-2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, spd_color, 1)

            # ── Violation flash ring ─────────────────
            if is_violation:
                cv2.circle(frame, (cx,cy), 18,
                    (0,0,255), 2)
                cv2.circle(frame, (cx,cy), 22,
                    (0,0,200), 1)
            if is_violation and track_id not in violations:
                violations[track_id] = True
                total_violations += 1
                print(f"  🚨 VIOLATION! ID:{track_id} "
                      f"{cls_name} → {kmh} KM/H")
                # Auto Screenshot
                violation_time = datetime.now().strftime("%H-%M-%S")

                screenshot_path = os.path.join(
                    OUTPUT_FOLDER,
                    f"Violation_ID_{track_id}_{cls_name}_{kmh}KMH_{violation_time}.jpg"
                )
                cv2.imwrite(screenshot_path, frame)

                print(f"  📸 Screenshot Saved: {screenshot_path}")
                #CSV FILE  
                csv_file = os.path.join(
                                        OUTPUT_FOLDER,
                                        "violations.csv"
                                    )

                with open(csv_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Time",
                        "Track ID",
                        "Vehicle",
                        "Speed(KMH)"
                    ])
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        track_id,
                        cls_name,
                        kmh
                    ])
            active_count += 1
    # ── HUD Panel ────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay,(0,0),(270,160),(0,0,0),-1)
    cv2.addWeighted(overlay,0.55,frame,0.45,0,frame)

    cv2.putText(frame, "TRINETRA  SPEED",
        (10,24), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0,212,200), 2)

    cv2.putText(frame,
        f"Frame       : {frame_count}/{total}",
        (10,48), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (200,200,200), 1)

    cv2.putText(frame,
        f"Active      : {active_count} vehicles",
        (10,66), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (0,255,180), 1)

    cv2.putText(frame,
        f"Speed Limit : {speed_limit} KM/H",
        (10,84), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (0,255,255), 1)

    cv2.putText(frame,
        f"Violations  : {total_violations}",
        (10,102),cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0,0,255) if total_violations > 0
        else (200,200,200), 1)

    cv2.putText(frame,
        f"Device      : {device.upper()}",
        (10,120),cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (200,200,200), 1)

    cv2.putText(frame,
        "Q=Quit  S=Screenshot  +/-=SpeedLimit",
        (10, height-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (80,80,80), 1)

    cv2.imshow("TRINETRA — Speed Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Stopped by user.")
        break
    elif key == ord('s'):
        screenshot_count += 1
        path = (f"D:\\shiva\\Projects\\TRINETRA\\"
                f"outputs\\speed_{screenshot_count}.jpg")
        cv2.imwrite(path, frame)
        print(f"  Screenshot saved: {path}")
    elif key == ord('+'):
        speed_limit += 5
        print(f"  Speed limit → {speed_limit} KM/H")
    elif key == ord('-'):
        speed_limit = max(10, speed_limit - 5)
        print(f"  Speed limit → {speed_limit} KM/H")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print(f"  Frames processed : {frame_count}")
print(f"  Speed violations : {total_violations}")
print(f"  Screenshots saved: {screenshot_count}")
print("\n  Day 4 COMPLETE — Speed detection working!")
print("=" * 50)