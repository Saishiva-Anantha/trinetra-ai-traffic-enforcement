from ultralytics import YOLO
import os

print("=" * 45)
print("  Downloading YOLOv8s model...")
print("=" * 45)

# This downloads ~22MB model to models folder
model = YOLO("yolov8s.pt")

# Move it to our models folder
import shutil
if os.path.exists("yolov8s.pt"):
    shutil.move("yolov8s.pt", 
                r"D:\shiva\Projects\TRINETRA\models\yolov8s.pt")
    print("\n Model saved to D:\\shiva\\Projects\\TRINETRA\\models\\")

print("\n Model info:")
print(f"  Task   : {model.task}")
print("\n YOLOv8s ready!")