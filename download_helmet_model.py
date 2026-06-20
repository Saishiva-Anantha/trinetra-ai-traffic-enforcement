import urllib.request
import os

print("Downloading helmet detection model...")

# We use a helmet-trained YOLOv8 model
url = ("https://github.com/ultralytics/assets/"
       "releases/download/v0.0.0/yolov8n.pt")

save_path = (r"D:\shiva\Projects\TRINETRA"
             r"\models\yolov8n.pt")

urllib.request.urlretrieve(url, save_path)
print(f"Saved to: {save_path}")
print("Done!")