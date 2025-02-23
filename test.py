from ultralytics import YOLO

model = YOLO("runs/detect/train5/weights/best.pt")

model.predict('input_videos/jack.mp4', save=True, show=True)