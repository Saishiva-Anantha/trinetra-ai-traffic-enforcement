import torch
import cv2
import ultralytics

print("=" * 45)
print("  TRAFFIC-EYE — Setup Verification")
print("=" * 45)

# Check PyTorch
print(f"\n PyTorch Version  : {torch.__version__}")

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f" GPU Detected     : {gpu_name}")
    print(f" VRAM Available   : {vram:.1f} GB")
    print(f" CUDA Version     : {torch.version.cuda}")
    print("\n  GPU is READY for TRAFFIC-EYE!")
else:
    print("\n  GPU NOT detected — running on CPU only")
    print("  Check CUDA installation.")

# Check OpenCV
print(f"\n OpenCV Version   : {cv2.__version__}")

# Check Ultralytics
print(f" Ultralytics      : {ultralytics.__version__}")

print("\n" + "=" * 45)
print("  Day 1 COMPLETE — All systems ready!")
print("=" * 45)