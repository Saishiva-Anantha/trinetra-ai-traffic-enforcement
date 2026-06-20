import cv2
from ultralytics import YOLO
import torch
import random
import numpy as np
from collections import defaultdict, deque

# ── Config ───────────────────────────────────────────
MODEL_PATH  = r"D:\shiva\Projects\TRINETRA\models\yolov8s.pt"
VIDEO_PATH  = r"D:\shiva\Projects\TRINETRA\videos\test_traffic.mp4"

TARGET_CLASSES = [0, 2, 3, 5, 7]
CLASS_NAMES    = {0:"Person", 2:"Car",
                  3:"Motorcycle", 5:"Bus", 7:"Truck"}

# ── Thresholds ───────────────────────────────────────
HELMET_HEAD_ZONE    = 0.45   # top 45% of moto box = head zone
TRIPLE_RIDE_COUNT   = 3      # persons on 1 moto = violation
OVERLAP_IOU_THRESH  = 0.15   # how much overlap counts
CONF_THRESHOLD      = 0.40
# ─────────────────────────────────────────────────────

print("=" * 52)
print("  TRINETRA — Helmet + Triple Riding Detector")
print("=" * 52)

model  = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device : {device.upper()}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Cannot open video!")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"  Video  : {width}x{height} @ {fps:.1f} FPS")
print("\n  Press Q=Quit | S=Screenshot")
print("=" * 52)

# ── Stats ─────────────────────────────────────────────
frame_count         = 0
screenshot_count    = 0
no_helmet_ids       = set()
triple_ride_ids     = set()
total_no_helmet     = 0
total_triple        = 0

def get_color(track_id):
    random.seed(int(track_id) * 3)
    return (random.randint(80, 255),
            random.randint(80, 255),
            random.randint(80, 255))

def calc_iou(boxA, boxB):
    """
    Calculate Intersection over Union of two boxes.
    Each box = (x1, y1, x2, y2)
    Returns float 0.0 → 1.0
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter_area

    return inter_area / union if union > 0 else 0.0

def check_helmet_in_zone(person_boxes, moto_box):
    """
    Check if any person's head is in the
    top 45% of the motorcycle bounding box.
    Returns True if helmet/person head found.
    """
    mx1, my1, mx2, my2 = moto_box
    head_zone_y2 = my1 + int((my2 - my1) * HELMET_HEAD_ZONE)
    head_zone    = (mx1, my1, mx2, head_zone_y2)

    for pb in person_boxes:
        iou = calc_iou(head_zone, pb)
        if iou > OVERLAP_IOU_THRESH:
            return True
    return False

def count_riders_on_moto(person_boxes, moto_box):
    """
    Count how many persons overlap with
    the motorcycle bounding box.
    """
    count = 0
    for pb in person_boxes:
        iou = calc_iou(moto_box, pb)
        if iou > OVERLAP_IOU_THRESH:
            count += 1
    return count

def draw_violation_banner(frame, x1, y1, x2,
                           text, color):
    """Draw a bold violation banner below the box."""
    banner_y1 = y2 + 4
    banner_y2 = y2 + 26
    cv2.rectangle(frame,
        (x1, banner_y1), (x2, banner_y2),
        color, -1)
    cv2.putText(frame, text,
        (x1 + 4, banner_y2 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("\nVideo ended.")
        break

    frame_count += 1

    # ── Run YOLO + ByteTrack ──────────────────────────
    results = model.track(
        frame,
        device=device,
        classes=TARGET_CLASSES,
        conf=CONF_THRESHOLD,
        iou=0.5,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )

    # ── Separate boxes by class ───────────────────────
    person_boxes     = []   # all person boxes this frame
    motorcycle_data  = []   # (track_id, x1,y1,x2,y2)
    all_detections   = []   # all detections for drawing

    for result in results:
        if result.boxes.id is None:
            continue

        for i, box in enumerate(result.boxes):
            track_id     = int(result.boxes.id[i])
            cls_id       = int(box.cls[0])
            conf         = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])

            all_detections.append(
                (track_id, cls_id, conf,
                 x1, y1, x2, y2))

            if cls_id == 0:   # Person
                person_boxes.append((x1,y1,x2,y2))
            elif cls_id == 3: # Motorcycle
                motorcycle_data.append(
                    (track_id, x1, y1, x2, y2))

    # ── Analyse each motorcycle ───────────────────────
    moto_violations = {}  # track_id → list of violations

    for (mid, mx1, my1, mx2, my2) in motorcycle_data:

        violations_this_moto = []

        # 1. Helmet check
        helmet_found = check_helmet_in_zone(
            person_boxes, (mx1, my1, mx2, my2))

        if not helmet_found:
            violations_this_moto.append("NO HELMET")
            if mid not in no_helmet_ids:
                no_helmet_ids.add(mid)
                total_no_helmet += 1
                print(f"  ⛑  NO HELMET! "
                      f"Motorcycle ID:{mid}")

        # 2. Triple riding check
        rider_count = count_riders_on_moto(
            person_boxes, (mx1, my1, mx2, my2))

        if rider_count >= TRIPLE_RIDE_COUNT:
            violations_this_moto.append(
                f"TRIPLE RIDING ({rider_count})")
            if mid not in triple_ride_ids:
                triple_ride_ids.add(mid)
                total_triple += 1
                print(f"  🏍  TRIPLE RIDING! "
                      f"ID:{mid} → {rider_count} riders")

        moto_violations[mid] = violations_this_moto

    # ── Draw all detections ───────────────────────────
    for (track_id, cls_id, conf,
         x1, y1, x2, y2) in all_detections:

        cls_name = CLASS_NAMES.get(cls_id, "?")
        is_moto  = (cls_id == 3)
        v_list   = moto_violations.get(track_id, [])
        has_viol = len(v_list) > 0

        # Box color
        if is_moto and has_viol:
            color = (0, 0, 255)      # red  = violation
        elif is_moto:
            color = (0, 200, 0)      # green = safe moto
        elif cls_id == 0:
            color = (0, 180, 255)    # orange = person
        else:
            color = get_color(track_id)

        # Bounding box
        cv2.rectangle(frame,
            (x1,y1),(x2,y2), color, 2)

        # Corner accents
        cl, tk = 10, 2
        cv2.line(frame,(x1,y1),(x1+cl,y1),color,tk)
        cv2.line(frame,(x1,y1),(x1,y1+cl),color,tk)
        cv2.line(frame,(x2,y1),(x2-cl,y1),color,tk)
        cv2.line(frame,(x2,y1),(x2,y1+cl),color,tk)
        cv2.line(frame,(x1,y2),(x1+cl,y2),color,tk)
        cv2.line(frame,(x1,y2),(x1,y2-cl),color,tk)
        cv2.line(frame,(x2,y2),(x2-cl,y2),color,tk)
        cv2.line(frame,(x2,y2),(x2,y2-cl),color,tk)

        # Main label
        label = f"ID:{track_id} {cls_name} {conf:.0%}"
        (lw,lh),_ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame,
            (x1, y1-lh-10),(x1+lw+8, y1),
            color, -1)
        cv2.putText(frame, label,
            (x1+4, y1-3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0,0,0), 1)

        # Violation banners under the box
        if is_moto:
            banner_y = y2 + 4
            for vtext in v_list:
                bw = x2 - x1
                cv2.rectangle(frame,
                    (x1, banner_y),
                    (x2, banner_y + 22),
                    (0,0,200), -1)
                cv2.putText(frame,
                    f"! {vtext}",
                    (x1+4, banner_y+15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255,255,255), 1)
                banner_y += 26

            # Safe badge
            if not has_viol and is_moto:
                cv2.rectangle(frame,
                    (x1, y2+4),
                    (x2, y2+22),
                    (0,140,0), -1)
                cv2.putText(frame,
                    "HELMET OK",
                    (x1+4, y2+16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255,255,255), 1)

        # Head zone indicator on motorcycles
        if is_moto:
            head_y = y1 + int((y2-y1)*HELMET_HEAD_ZONE)
            cv2.line(frame,
                (x1, head_y),(x2, head_y),
                (0,255,255), 1)

    # ── HUD Panel ─────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay,(0,0),(285,175),(0,0,0),-1)
    cv2.addWeighted(overlay,0.55,frame,0.45,0,frame)

    cv2.putText(frame, "TRINETRA  SAFETY",
        (10,24), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0,212,200), 2)

    cv2.putText(frame,
        f"Frame        : {frame_count}/{total}",
        (10,48), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (200,200,200), 1)

    cv2.putText(frame,
        f"Motorcycles  : {len(motorcycle_data)}",
        (10,66), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (0,255,180), 1)

    cv2.putText(frame,
        f"No Helmet    : {total_no_helmet}",
        (10,84), cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0,0,255) if total_no_helmet > 0
        else (200,200,200), 1)

    cv2.putText(frame,
        f"Triple Riding: {total_triple}",
        (10,102), cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0,0,255) if total_triple > 0
        else (200,200,200), 1)

    cv2.putText(frame,
        f"Persons      : {len(person_boxes)}",
        (10,120), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (0,180,255), 1)

    cv2.putText(frame,
        f"Device       : {device.upper()}",
        (10,138), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (200,200,200), 1)

    cv2.putText(frame,
        "Q=Quit  S=Screenshot",
        (10, height-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (80,80,80), 1)

    cv2.imshow("TRINETRA — Safety Check", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Stopped by user.")
        break
    elif key == ord('s'):
        screenshot_count += 1
        path = (f"D:\\shiva\\Projects\\TRINETRA\\"
                f"outputs\\safety_{screenshot_count}.jpg")
        cv2.imwrite(path, frame)
        print(f"  Screenshot saved: {path}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 52)
print(f"  Frames processed  : {frame_count}")
print(f"  No helmet cases   : {total_no_helmet}")
print(f"  Triple riding     : {total_triple}")
print(f"  Screenshots saved : {screenshot_count}")
print("\n  Day 5 COMPLETE — Safety detection working!")
print("=" * 52)