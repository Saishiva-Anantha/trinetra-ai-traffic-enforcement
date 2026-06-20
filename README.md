\# 🚦 TRINETRA — AI-Powered Traffic Enforcement System



TRINETRA is a real-time computer vision system that detects traffic violations

from CCTV footage — speeding, no-helmet riding, triple riding, and (upcoming)

red-light jumping and automatic license plate recognition.



Built as a personal deep-learning project to learn real-time object detection,

multi-object tracking, and applied computer vision.



\---



\## 🎯 Features Implemented So Far



\- ✅ Real-time vehicle detection (YOLOv8)

\- ✅ Persistent multi-object tracking with unique IDs (ByteTrack)

\- ✅ Speed estimation in KM/H using pixel-to-meter calibration

\- ✅ Helmet violation detection (head-zone IoU analysis)

\- ✅ Triple riding detection (rider overlap counting)



\## 🔜 Coming Soon



\- ⬜ Red light violation detection

\- ⬜ Automatic Number Plate Recognition (ANPR)

\- ⬜ PyQt6 desktop application (Highway / Urban modes)

\- ⬜ SQLite database + SMS alerts + auto-generated PDF challans



\---



\## 🧠 Tech Stack



| Component       | Technology                  |

|------------------|------------------------------|

| Detection Model  | YOLOv8 (Ultralytics)        |

| Tracking         | ByteTrack                   |

| Framework        | PyTorch (CUDA 11.8)         |

| Computer Vision  | OpenCV                      |

| Language         | Python 3.10                 |

| Hardware         | NVIDIA GTX 1650 (4GB VRAM)  |



\---



\## 📁 Project Structure



\\`\\`\\`

TRINETRA/

├── src/              # All Python source code

│   ├── day2\_detection.py

│   ├── day3\_tracking.py

│   ├── day4\_speed.py

│   └── day5\_helmet.py

├── models/           # YOLO weights (not pushed — see below)

├── videos/           # Test footage (not pushed — see below)

├── outputs/          # Saved screenshots/results

└── README.md

\\`\\`\\`



\---



\## ⚙️ Setup Instructions



\\`\\`\\`bash

\# Clone the repo

git clone https://github.com/YOUR\_USERNAME/trinetra-ai-traffic-enforcement.git

cd trinetra-ai-traffic-enforcement



\# Create virtual environment (Python 3.10)

py -3.10 -m venv venv

venv\\\\Scripts\\\\activate



\# Install dependencies

pip install -r requirements.txt

\\`\\`\\`



\### Model Weights

YOLOv8 weights are not included in this repo (file size). They auto-download

on first run via the \\`ultralytics\\` package, or manually:

\\`\\`\\`python

from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # auto-downloads

\\`\\`\\`



\### Test Video

Place any traffic CCTV footage (MP4) inside \\`videos/test\_traffic.mp4\\`.



\---



\## 🚀 Running the Project



\\`\\`\\`bash

\# Day 2 — Basic detection

python src/day2\_detection.py



\# Day 3 — Tracking with unique IDs

python src/day3\_tracking.py



\# Day 4 — Speed detection

python src/day4\_speed.py



\# Day 5 — Helmet + triple riding detection

python src/day5\_helmet.py

\\`\\`\\`



\---



\## 📊 Development Log



| Day | Milestone                                      |

|-----|-------------------------------------------------|

| 1   | Environment setup — PyTorch + CUDA + GPU verified |

| 2   | First YOLOv8 vehicle detection on video          |

| 3   | ByteTrack integration — persistent vehicle IDs   |

| 4   | Speed calculation (pixel-to-meter + KM/H)        |

| 5   | Helmet detection + triple riding logic           |



\---



\## 👤 Author



\*\*Shiva\*\*

Built as a hands-on deep-learning project — first computer vision system.



\---



\## 📄 License



This project is for educational and portfolio purposes.

